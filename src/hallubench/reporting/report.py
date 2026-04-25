"""Reproducible Markdown+HTML report rendered from the saved summary only."""
from pathlib import Path
from html import escape
from hallubench.storage.results import read_json
from hallubench.reporting.plots import create_plots


def generate_report(directory):
    directory = Path(directory)
    summary = read_json(directory/"summary.json")
    manifest = read_json(directory/"experiment.json")
    create_plots(directory, summary, manifest)
    lines = ["# HalluBench — Findings Report", "", f"Experiment status: **{manifest['status']}**",
             f"Held-out examples selected: **{manifest['n_test']}**",
             f"Evaluation tier: **{manifest.get('evaluation_tier', 'unknown')}** (fixture smoke is NOT a public benchmark)",
             f"Test-ID SHA-256: `{manifest['test_ids_sha256']}`", "",
             "## Objective and interpretation", "",
             "Measures agreement between detector predictions and dataset-defined, evidence-relative labels.",
             "It does not establish open-world truth, claim completeness, or detector reliability on unseen domains.",
             "HaluEval 'hallucinated' is a constructed coarse label, not an adjudicated contradicted-claim label.",
             "", "## Observed performance", "",
             "| Detector | Evaluated | Coverage | Hallucination precision | Recall | F1 | Accuracy | Mean latency (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, block in summary["detectors"].items():
        r, m = block["aggregate"], block["aggregate"]["metrics"]
        cell = (lambda x: f"{x:.4f}" if x is not None else "N/A")
        lines.append(f"| {name} | {r['evaluated']} | {cell(r['coverage'])} | {cell(m.get('precision') if m else None)} | "
                     f"{cell(m.get('recall') if m else None)} | {cell(m.get('f1') if m else None)} | "
                     f"{cell(m.get('accuracy') if m else None)} | {block['mean_latency_ms']:.2f} |")
    lines += ["", "## Data eligibility and failure modes", ""]
    for name, block in summary["detectors"].items():
        r = block["aggregate"]
        lines += [f"### {name}", f"- Eligible reference labels: {r['eligible']}; correctly excluded unknown-reference labels: {r['excluded_truth_insufficient']}.",
                  f"- Processing status counts: `{r['statuses']}`; coverage uses eligible denominator, never converts failures into predictions.",
                  f"- Observed false positives/negatives in stored errors: {block['errors']}; suggested categories need human review.",
                  f"- New-inference wall time: {block['wall_seconds_new_inferences']:.3f}s (resumed runs exclude reused inferences)."]
        for field, groups in (("task", block["by_task"]), ("dataset", block["by_dataset"]), ("generator", block["by_generator"])):
            lines.append(f"- By {field}: " + ", ".join(f"{k}={v['metrics']['f1']:.3f} F1 (n={v['evaluated']})" if v["metrics"] else f"{k}=N/A" for k,v in groups.items()))
    lines += ["", "## Paired uncertainty (grouped bootstrap)", ""]
    for name, value in summary["paired"].items():
        lines.append(f"- {name}: {value}")
    lines += ["", "## Generated artifacts", "",
              "- `summary.json`, `experiment.json`, `dataset_quality.json`: experiment metadata and quantitative results.",
              "- `predictions_*.jsonl`, `errors_*.jsonl`: all returned statuses and stored errors.",
              "- `confusion_*.png`, `detector_comparison.png`, `latency.png`; optional reliability, score distribution, PR curve and descriptive threshold-sensitivity plots.",
              "", "## Limits and required next research", "",
              "- Unsupported is not logically false; neutral NLI outputs cannot prove incorrectness.",
              "- Response labels are not automatically atomic-claim labels, and sentence splitting is approximate.",
              "- API/self-reported confidence is not calibrated probability; no costs are inferred without usage/prices.",
              "- Dataset construction artifacts, distribution shift and correlated sources constrain generalization.",
              "- Bootstrap intervals are exploratory; significance and universal superiority are NOT claimed.",
              "- Missing model or data access produces no evaluation rather than a zero score."]
    md = "\n".join(lines) + "\n"
    (directory/"findings.md").write_text(md, encoding="utf-8")
    html = "<!doctype html><html><head><meta charset='utf-8'><title>HalluBench findings</title>" \
           "<style>body{font:16px/1.5 system-ui;max-width:940px;margin:48px auto;padding:0 20px}pre{white-space:pre-wrap}img{max-width:100%}</style></head><body>" \
           "<h1>HalluBench findings</h1><pre>" + escape(md) + "</pre>" \
           "<img src='detector_comparison.png'><img src='latency.png'></body></html>"
    (directory/"findings.html").write_text(html, encoding="utf-8")
    return directory/"findings.md"
