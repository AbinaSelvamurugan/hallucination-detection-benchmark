"""CLI: prepare -> validate -> train/tune -> run -> report; no hidden downloads."""
import argparse
from pathlib import Path
from hallubench.config import read_config, fingerprint, sha256_file
from hallubench.schemas import Example, binary_truth
from hallubench.storage.results import write_json, write_jsonl, read_json, read_jsonl
from hallubench.datasets.hf_loader import normalize_halu, normalize_ragtruth_local, load_hf, load_local, revision_for_local
from hallubench.datasets.normalization import explicit_mapping_rows
from hallubench.datasets.splitting import group_split
from hallubench.datasets.validation import quality_report
from hallubench.evaluation.runner import load_prepared, evaluate
from hallubench.detectors.registry import create


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hallubench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Inspect explicit Hugging Face config/schema")
    p.add_argument("--repo", required=True); p.add_argument("--config", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--split", default="data")
    p.add_argument("--limit", type=int, default=2)
    p = sub.add_parser("prepare", help="Normalize source and grouped split")
    p.add_argument("--source", choices=["halu", "ragtruth", "local", "hf-mapped"], required=True)
    p.add_argument("--task", choices=["qa", "dialogue", "summarization"])
    p.add_argument("--revision"); p.add_argument("--repo"); p.add_argument("--config"); p.add_argument("--split", default="train")
    p.add_argument("--input"); p.add_argument("--sources"); p.add_argument("--mapping")
    p.add_argument("--limit", type=int); p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--holdout-generator", action="append")
    p = sub.add_parser("validate"); p.add_argument("--prepared", required=True)
    p = sub.add_parser("train"); p.add_argument("--prepared", required=True); p.add_argument("--checkpoint", required=True)
    p = sub.add_parser("tune"); p.add_argument("--prepared", required=True)
    p.add_argument("--detector", choices=["heuristic", "semantic"], required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("run"); p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True); p.add_argument("--detectors", nargs="+", default=["heuristic"])
    p.add_argument("--config", help="Detector kwargs in YAML"); p.add_argument("--task", action="append")
    p.add_argument("--max-examples", type=int); p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p = sub.add_parser("claims"); p.add_argument("--prepared", required=True); p.add_argument("--output", required=True)
    p.add_argument("--detector", choices=["heuristic", "semantic", "nli", "classifier"], default="heuristic")
    p.add_argument("--checkpoint"); p.add_argument("--max-examples", type=int)
    p = sub.add_parser("report"); p.add_argument("--results", required=True)
    p = sub.add_parser("compare"); p.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        rows = list(load_hf(args.repo, args.config, args.split, args.revision, limit=args.limit))
        print({"repo": args.repo, "config": args.config, "split": args.split, "revision": args.revision,
               "rows_previewed": len(rows), "columns": list(rows[0]) if rows else []})
    elif args.command == "prepare":
        if args.source == "halu":
            if not args.task or not args.revision:
                parser.error("halu requires --task --revision")
            examples = list(normalize_halu(args.repo or "pminervini/HaluEval", args.task, args.revision, limit=args.limit))
        elif args.source == "ragtruth":
            if not args.input or not args.sources:
                parser.error("ragtruth requires --input response.jsonl --sources source_info.jsonl")
            revision = args.revision or fingerprint({"response_sha256": sha256_file(args.input), "source_sha256": sha256_file(args.sources)})
            examples = list(normalize_ragtruth_local(args.input, args.sources, revision=revision))
            if args.limit: examples = examples[:args.limit]
        else:
            if not args.mapping or (args.source == "local" and not args.input):
                parser.error("mapped sources require --mapping; local additionally requires --input")
            cfg = read_config(args.mapping)
            if args.source == "local":
                revision = revision_for_local(args.input)
                rows = load_local(args.input)
            else:
                if not (args.repo and args.config and args.revision):
                    parser.error("hf-mapped requires --repo --config --revision")
                revision = args.revision
                rows = load_hf(args.repo, args.config, args.split, revision, limit=args.limit)
            from itertools import islice
            examples = list(explicit_mapping_rows(islice(rows, args.limit) if args.limit and args.source == "local" else rows,
                               cfg, args.repo or args.input, revision))
        group_split(examples, seed=args.seed, holdout_generators=set(args.holdout_generator or []))
        quality = quality_report(examples)
        report_path = str(args.output) + ".quality.json"
        write_json(report_path, {**quality, "source": args.source, "seed": args.seed,
                                 "holdout_generators": args.holdout_generator})
        if not quality["pass"]:
            raise ValueError(f"Dataset quality failed; inspect {report_path}: {str(quality)[:700]}")
        write_jsonl(args.output, [e.to_dict() for e in examples])
        print(f"Saved {len(examples)} verified-schema rows and quality report to {args.output}")
    elif args.command == "validate":
        quality = quality_report(load_prepared(args.prepared))
        print(quality)
        if not quality["pass"]: raise SystemExit(1)
    elif args.command == "train":
        from hallubench.detectors.classifier import TrainedClassifier
        examples = load_prepared(args.prepared)
        if not quality_report(examples)["pass"]: raise ValueError("Quality validation failed")
        clf = TrainedClassifier()
        info = clf.train([x for x in examples if x.split == "train"], [x for x in examples if x.split == "validation"])
        clf.save(args.checkpoint)
        print(info)
    elif args.command == "tune":
        examples = load_prepared(args.prepared)
        validation = [x for x in examples if x.split == "validation" and binary_truth(x) is not None]
        if not validation: raise ValueError("No labelled validation data")
        from hallubench.evaluation.metrics import select_threshold
        detector = create(args.detector, threshold=.5)
        predictions = [detector.predict(e) for e in validation]
        key = "lexical_overlap" if args.detector == "heuristic" else "cosine_similarity"
        selected = [(binary_truth(ex), p.scores[key]) for ex, p in zip(validation, predictions) if p.status == "ok" and key in p.scores]
        if not selected: raise ValueError("No valid validation scores")
        # detector positive iff similarity < threshold; convert to a hallucination score.
        thresholds = [x/100 for x in range(5, 100, 5)]
        t = select_threshold([v[0] for v in selected], [1-v[1] for v in selected], thresholds)
        tuned = 1-t
        result = {"detector": args.detector, "threshold": tuned, "n_validation": len(selected),
                  "objective": "validation positive F1", "test_touched": False,
                  "validation_ids_sha256": fingerprint([x.example_id for x in validation])}
        write_json(args.output, result)
        print(result)
        detector.close()
    elif args.command == "run":
        cfg = read_config(args.config) if args.config else {}
        detectors = [create(n, **cfg.get(n, {})) for n in args.detectors]
        if len(set(x.name for x in detectors)) != len(detectors):
            raise ValueError("Duplicate detectors are not permitted")
        try:
            out = evaluate(args.prepared, detectors, args.output, tasks=args.task,
                           max_examples=args.max_examples, resume=args.resume, batch_size=args.batch_size)
            print({d: b["aggregate"] for d,b in out.items()})
        finally:
            for detector in detectors: detector.close()
    elif args.command == "claims":
        from hallubench.claim_analysis.segmentation import predicted_error_spans
        from hallubench.claim_analysis.span_matching import span_metrics
        test = [e for e in load_prepared(args.prepared) if e.split == "test" and e.annotations]
        if args.max_examples is not None: test = test[:args.max_examples]
        if not test: raise ValueError("No annotated test spans available (RAGTruth or explicit local annotations)")
        detector = create(args.detector, **({"checkpoint": args.checkpoint} if args.detector == "classifier" else {}))
        items, output = [], []
        try:
            for ex in test:
                pred, records = predicted_error_spans(detector, ex)
                items.append((ex.annotations, pred))
                output.append({"parent_id": ex.example_id, "gold": [x.__dict__ for x in ex.annotations],
                               "predicted": [x.__dict__ for x in pred], "sentence_records": records})
        finally:
            detector.close()
        write_jsonl(args.output, output)
        result = span_metrics(items)
        result["warning"] = "Sentence proxy spans, not validated atomic claim extraction; interpret span recall cautiously"
        write_json(str(args.output)+".metrics.json", result)
        print(result)
    elif args.command == "report":
        from hallubench.reporting.report import generate_report
        print(generate_report(args.results))
    elif args.command == "compare":
        for name, block in read_json(Path(args.results)/"summary.json")["detectors"].items():
            print(name, block["aggregate"])

if __name__ == "__main__":
    main()
