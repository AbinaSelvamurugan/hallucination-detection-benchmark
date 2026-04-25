"""Public Hub + local dataset loading; pinned revisions and explicit schema adapters."""
from __future__ import annotations
import csv
from pathlib import Path
from hallubench.datasets.normalization import halu_examples, ragtruth_examples, explicit_mapping_rows
from hallubench.storage.results import read_jsonl
from hallubench.config import sha256_file


def load_local(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as file:
            yield from csv.DictReader(file)
    else:
        raise ValueError("Local file must be .csv or .jsonl")


def load_hf(repo: str, config: str, split: str, revision: str, limit: int | None = None,
            streaming: bool = False, cache_dir: str | None = None):
    if not all((repo, config, split, revision)):
        raise ValueError("Hugging Face repo/config/split/revision must be specified")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the hf extra: pip install -e '.[hf]'") from exc
    ds = load_dataset(repo, name=config, split=split, revision=revision,
                      streaming=streaming, cache_dir=cache_dir, trust_remote_code=False)
    for n, row in enumerate(ds):
        if limit is not None and n >= limit:
            break
        yield row


def normalize_halu(repo: str, task: str, revision: str, limit: int | None = None, **kw):
    if repo != "pminervini/HaluEval":
        raise ValueError("HaluEval adapter is verified only for pminervini/HaluEval")
    for n, row in enumerate(load_hf(repo, task, "data", revision, limit=limit, **kw)):
        yield from halu_examples(row, n, task, revision)


def normalize_ragtruth_local(responses, sources, revision="local-unpinned"):
    """Join official response.jsonl and source_info.jsonl; accept source id or source_id."""
    lookup = {str(r.get("source_id", r.get("id"))): r for r in load_local(sources)}
    for response in load_local(responses):
        key = str(response.get("source_id"))
        if key not in lookup:
            raise ValueError(f"RAGTruth orphaned response source_id={key}")
        yield from ragtruth_examples(response, lookup[key], revision)


def revision_for_local(path):
    return "sha256:" + sha256_file(path)
