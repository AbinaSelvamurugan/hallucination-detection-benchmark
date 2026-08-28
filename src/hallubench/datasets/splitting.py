"""Response/source grouped splits; a parent response never straddles splits."""
from __future__ import annotations
import hashlib
import re
from collections import defaultdict
from hallubench.schemas import Example


def normalized_text(s):
    return " ".join(re.findall(r"\w+", (s or "").casefold()))


def group_split(examples: list[Example], seed=17, train=0.7, validation=0.15,
                preserve_existing=True, holdout_generators: set[str] | None = None,
                carve_validation_from_train=True) -> list[Example]:
    if train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("Split fractions must allocate non-empty train, validation and test")
    groups = defaultdict(list)
    # Union-find: group by underlying source AND identical normalized question across datasets.
    parents = list(range(len(examples)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parents[pb] = pa
    seen = {}
    for idx, ex in enumerate(examples):
        for key in (("grp", ex.group_id),
                    ("question", normalized_text(ex.question) if normalized_text(ex.question) not in ("summarize the evidence", "use the provided evidence to answer the task") else ""),
                    ("source_text", normalized_text(ex.evidence) if len(normalized_text(ex.evidence)) >= 80 else ""),
                    # same rule as cross_split_collisions: identical long responses must share a split
                    ("answer", normalized_text(ex.answer) if len(normalized_text(ex.answer)) >= 40 else "")):
            if key[1]:
                if key in seen:
                    union(idx, seen[key])
                else:
                    seen[key] = idx
    for idx, ex in enumerate(examples):
        groups[find(idx)].append(ex)
    for group in groups.values():
        fixed = {ex.split for ex in group if ex.split != "unspecified"}
        if preserve_existing and len(fixed) > 1:
            raise ValueError("Source/question crosses upstream splits; cannot claim leakage-free fixed splits")
        fixed_split = next(iter(fixed)) if preserve_existing and fixed else None
        generators = {x.generator for x in group}
        if holdout_generators and any(g in holdout_generators for g in generators):
            if fixed_split and fixed_split != "test":
                raise ValueError("Generator holdout conflicts with fixed upstream split")
            fixed_split = "test"
        digest = hashlib.sha256(f"{seed}:{min(x.group_id for x in group)}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        selection = fixed_split or ("train" if u < train else "validation" if u < train + validation else "test")
        if fixed_split == "train" and carve_validation_from_train:
            # Preserve official test, carve a deterministic DEV validation subset of upstream train.
            selection = "validation" if u < validation / (train + validation) else "train"
        for ex in group:
            ex.split = selection
    return examples


def cross_split_collisions(examples, answer_similarity=0.97):
    """Check group, question, exact response; near-duplicate response using char TF-IDF."""
    issues = []
    for name, fn in (("group", lambda x: x.group_id),
                     ("question", lambda x: normalized_text(x.question)),
                     ("answer", lambda x: normalized_text(x.answer)),
                     ("source_text", lambda x: normalized_text(x.evidence))):
        encountered = {}
        for ex in examples:
            key = fn(ex)
            if name == "question" and key in ("summarize the evidence", "use the provided evidence to answer the task"):
                continue
            if name == "source_text" and len(key) < 80:
                continue
            if name == "answer" and len(key) < 40:
                continue
            if not key:
                continue
            if key in encountered and encountered[key] != ex.split:
                issues.append(f"{name} split collision: {key[:80]}")
            encountered[key] = ex.split
    if 1 < len(examples) <= 3000:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        texts = [normalized_text(x.answer) for x in examples]
        if len(set(texts)) > 1:
            x = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(texts)
            # Bounded nearest-neighbor query avoids a dense NxN matrix.
            neigh = NearestNeighbors(n_neighbors=min(6, len(examples)), metric="cosine", algorithm="brute").fit(x)
            distances, ids = neigh.kneighbors(x)
            for i, (ds, near) in enumerate(zip(distances, ids)):
                for dist, j in zip(ds, near):
                    if i < j and len(texts[i]) >= 40 and len(texts[j]) >= 40 and dist <= 1 - answer_similarity and examples[i].split != examples[j].split:
                        issues.append(f"near-duplicate answer: {examples[i].example_id} <> {examples[j].example_id}")
    # Above 3,000 records exact duplicates remain checked; approximate checking
    # must be run separately or sharded by task/source for full-scale validation.
    return issues
