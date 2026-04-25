"""Record validation and pre-experiment dataset quality diagnostics."""
from __future__ import annotations
from collections import Counter
from hallubench.datasets.splitting import normalized_text, cross_split_collisions


def quality_report(examples, check_leakage=True):
    invalid, missing_evidence, ids, exact, duplicated, question_counts = [], 0, set(), set(), [], Counter()
    for ex in examples:
        try:
            ex.validate()
        except (ValueError, TypeError) as e:
            invalid.append({"id": ex.example_id, "reason": str(e)})
        if not ex.evidence or not str(ex.evidence).strip():
            missing_evidence += 1
        if ex.example_id in ids:
            duplicated.append(ex.example_id)
        ids.add(ex.example_id)
        key = (ex.dataset, normalized_text(ex.question), normalized_text(ex.answer))
        if key in exact:
            duplicated.append(ex.example_id)
        exact.add(key)
        question_counts[normalized_text(ex.question)] += 1
    leakage = cross_split_collisions(examples) if check_leakage else []
    return {"n_examples": len(examples), "labels": dict(Counter(x.label for x in examples)),
        "tasks": dict(Counter(x.task for x in examples)), "splits": dict(Counter(x.split for x in examples)),
        "datasets": dict(Counter(x.dataset for x in examples)),
        "missing_evidence": missing_evidence, "with_annotations": sum(bool(x.annotations) for x in examples),
        "invalid": invalid, "duplicates": duplicated,
        "questions_repeated": sum(v > 1 for k, v in question_counts.items() if k),
        "leakage_issues": leakage,
        "near_duplicate_scan": "char TF-IDF nearest neighbor" if len(examples) <= 3000 else
                              "exact checks only (>3000 records)",
        "warnings": (["Full near-duplicate audit required before publication"] if len(examples)>3000 else []),
        "pass": not bool(invalid or duplicated or leakage) and bool(examples)}
