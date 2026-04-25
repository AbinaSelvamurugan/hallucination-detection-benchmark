"""Same-test runner, atomic persistence, abstention accounting and subgroups."""
from __future__ import annotations
import time
from collections import defaultdict
from pathlib import Path
from hallubench.schemas import Example
from hallubench.storage.results import read_jsonl, write_json, write_jsonl
from hallubench.datasets.validation import quality_report
from hallubench.evaluation.metrics import compute_metrics
from hallubench.evaluation.error_analysis import error_rows
from hallubench.config import fingerprint, sha256_file


def load_prepared(path):
    examples = [Example.from_dict(r) for r in read_jsonl(path)]
    for e in examples: e.validate()
    return examples


def evaluate(prepared, detectors, output, tasks=None, max_examples=None, resume=False, batch_size=16):
    data = load_prepared(prepared)
    quality = quality_report(data)
    if not quality["pass"]:
        raise ValueError(f"Quality check failed: {quality['invalid'][:2]}, {quality['duplicates'][:2]}, {quality['leakage_issues'][:2]}")
    eligible = [x for x in data if x.split == "test" and (tasks is None or x.task in tasks)]
    eligible = sorted(eligible, key=lambda x: x.example_id)
    if max_examples is not None:
        eligible = eligible[:max_examples]
    if not eligible:
        raise ValueError("No test examples; test data never constructed synthetically")
    base = Path(output)
    base.mkdir(parents=True, exist_ok=True)
    write_json(base/"dataset_quality.json", quality)
    manifest = {"prepared_file": str(prepared), "prepared_sha256": sha256_file(prepared), "dataset_revision": sorted(set((e.dataset, e.revision) for e in eligible)),
        "n_test": len(eligible), "test_ids_sha256": fingerprint([e.example_id for e in eligible]),
        "tasks": tasks, "max_examples": max_examples, "batch_size": batch_size, "status": "running",
        "evaluation_tier": "fixture_smoke" if all(e.dataset.startswith("TOY") for e in eligible) else "labelled_dataset",
        "detectors": {d.name: d.version for d in detectors}}
    existing_manifest = base/"experiment.json"
    if resume and existing_manifest.exists():
        from hallubench.storage.results import read_json
        old = read_json(existing_manifest)
        if any(old.get(k) != manifest[k] for k in ("test_ids_sha256", "prepared_sha256", "detectors", "batch_size")):
            raise ValueError("Cannot resume with different test IDs")
    write_json(existing_manifest, manifest)
    summaries, all_predictions = {}, {}
    for detector in detectors:
        path = base/f"predictions_{detector.name}.jsonl"
        previous = {}
        if resume and path.exists():
            from hallubench.schemas import Prediction
            previous = {row["example_id"]: Prediction(**row) for row in read_jsonl(path)}
        to_run = [e for e in eligible if e.example_id not in previous]
        start = time.perf_counter()
        fresh = detector.predict_batch(to_run, batch_size=batch_size)
        inference_seconds = time.perf_counter()-start
        preds = {**previous, **{p.example_id: p for p in fresh}}
        if len(preds) != len(eligible) or set(preds) != {x.example_id for x in eligible}:
            raise ValueError("Detector outputs incomplete or mismatched IDs")
        write_jsonl(path, [preds[e.example_id].to_dict() for e in eligible])
        all_predictions[detector.name] = preds
        metrics = compute_metrics(eligible, preds)
        by_task = {}
        for task in sorted({e.task for e in eligible}):
            group = [e for e in eligible if e.task == task]
            by_task[task] = compute_metrics(group, preds)
        by_dataset = {}
        for source in sorted({e.dataset for e in eligible}):
            by_dataset[source] = compute_metrics([e for e in eligible if e.dataset == source], preds)
        by_generator = {}
        for name in sorted({e.generator or "unknown" for e in eligible}):
            by_generator[name] = compute_metrics([e for e in eligible if (e.generator or "unknown") == name], preds)
        errors = error_rows(eligible, preds)
        write_jsonl(base/f"errors_{detector.name}.jsonl", errors)
        summaries[detector.name] = {"aggregate": metrics, "by_task": by_task,
             "by_dataset": by_dataset, "by_generator": by_generator,
             "errors": len(errors), "n_new_inferences": len(fresh),
             "wall_seconds_new_inferences": inference_seconds,
             "mean_latency_ms": sum(p.latency_ms for p in preds.values())/len(preds),
             "version": detector.version}
    # Paired intersection includes only the cases BOTH methods could evaluate.
    from hallubench.evaluation.statistics import paired_bootstrap
    pairwise = {}
    for i, a in enumerate(detectors):
        for b in detectors[i+1:]:
            pairwise[f"{a.name}__vs__{b.name}"] = paired_bootstrap(eligible, all_predictions[a.name],
                                                                      all_predictions[b.name], iterations=250)
    write_json(base/"summary.json", {"detectors": summaries, "paired": pairwise})
    manifest["status"] = "finished_with_detector_errors" if any(
        d["aggregate"]["statuses"]["error"] or d["aggregate"]["statuses"]["unavailable"]
        for d in summaries.values()) else "finished"
    manifest["detectors"] = {d.name: d.version for d in detectors}
    write_json(existing_manifest, manifest)
    return summaries
