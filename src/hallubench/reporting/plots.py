"""Plots computed strictly from persisted evaluation outputs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plots(directory, summary, manifest=None):
    out = Path(directory)
    names, scores, latencies = [], [], []
    for name, block in summary["detectors"].items():
        m = block["aggregate"]["metrics"]
        if m is not None:
            names.append(name)
            scores.append([m.get("precision", np.nan), m.get("recall", np.nan),
                           m.get("f1", np.nan), m["accuracy"]])
            latencies.append(block["mean_latency_ms"])
            cm = np.array(m["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            ax.set_xticks(range(len(cm)), m["confusion_labels"], rotation=25)
            ax.set_yticks(range(len(cm)), m["confusion_labels"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
            ax.set_title(f"{name}: evaluated predictions only")
            fig.tight_layout(); fig.savefig(out/f"confusion_{name}.png", dpi=160); plt.close(fig)
            cal = m.get("calibration")
            if cal:
                fig, ax = plt.subplots(figsize=(5, 4))
                data = cal["reliability"]
                ax.plot([0,1],[0,1], ls="--", color="grey")
                ax.scatter([x["mean_predicted"] for x in data], [x["observed_frequency"] for x in data])
                ax.set(xlabel="Mean model probability", ylabel="Empirical positive rate",
                       title=f"{name}: uncalibrated reliability", xlim=(0,1), ylim=(0,1))
                fig.tight_layout(); fig.savefig(out/f"reliability_{name}.png", dpi=160); plt.close(fig)
    # Diagnostic score plots only when the original prepared data and
    # per-example probability scores are both available. Test thresholds here
    # are descriptive curves: they are NEVER used to choose a final threshold.
    if manifest:
        from hallubench.schemas import Prediction, binary_truth
        from hallubench.evaluation.metrics import binary_prob
        from hallubench.evaluation.runner import load_prepared
        from hallubench.storage.results import read_jsonl
        from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
        prepared = Path(manifest.get("prepared_file", ""))
        if prepared.exists():
            examples = {e.example_id: e for e in load_prepared(prepared) if e.split == "test"}
            for name in summary["detectors"]:
                saved = out / f"predictions_{name}.jsonl"
                if not saved.exists():
                    continue
                pairs = []
                for row in read_jsonl(saved):
                    ex = examples.get(row["example_id"])
                    if ex is None or binary_truth(ex) is None:
                        continue
                    score = binary_prob(Prediction(**row))
                    if score is not None:
                        pairs.append((binary_truth(ex), score))
                if len(pairs) < 2 or len(set(y for y,p in pairs)) < 2:
                    continue
                yt = np.array([x[0] for x in pairs], dtype=int)
                pr = np.array([x[1] for x in pairs], dtype=float)
                fig, ax = plt.subplots(figsize=(6,4))
                for klass, desc in ((0,"Supported"),(1,"Hallucination label")):
                    ax.hist(pr[yt == klass], bins=np.linspace(0,1,11), alpha=.6, label=desc)
                ax.set(xlabel="Model probability of hallucination", ylabel="Count",
                       title=f"{name}: test score distributions")
                ax.legend(); fig.tight_layout(); fig.savefig(out / f"score_distribution_{name}.png",dpi=160);plt.close(fig)
                precision, recall, _ = precision_recall_curve(yt, pr)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.plot(recall,precision,marker=".")
                ax.set(xlim=(0,1),ylim=(0,1),xlabel="Hallucination recall",
                       ylabel="Hallucination precision",title=f"{name}: test PR curve (descriptive)")
                fig.tight_layout();fig.savefig(out/f"pr_curve_{name}.png",dpi=160);plt.close(fig)
                thresholds = np.linspace(0.05,.95,19)
                sweep = [precision_recall_fscore_support(yt, pr >= t, average="binary", zero_division=0)
                         for t in thresholds]
                fig,ax=plt.subplots(figsize=(6,4))
                for i,label in enumerate(("precision","recall","F1")):
                    ax.plot(thresholds,[x[i] for x in sweep],label=label)
                ax.set(xlabel="Threshold (not fitted on test)",ylabel="Score",ylim=(0,1),
                       title=f"{name}: descriptive sensitivity")
                ax.legend();fig.tight_layout();fig.savefig(out/f"threshold_sensitivity_{name}.png",dpi=160);plt.close(fig)
    if scores:
        x = np.arange(len(names))
        vals = np.asarray(scores)
        fig, ax = plt.subplots(figsize=(max(7, len(names)*1.5), 4.5))
        width = .2
        for i, label in enumerate(("Precision", "Recall", "F1", "Accuracy")):
            ax.bar(x+(i-1.5)*width, vals[:,i], width, label=label)
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("Score (evaluated subset)")
        ax.legend(loc="lower right"); fig.tight_layout()
        fig.savefig(out/"detector_comparison.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(max(6, len(names)*1.5),4))
        ax.bar(names, latencies)
        ax.set_ylabel("Mean inference latency (ms)")
        ax.set_title("Measured latency (may include model load on first call)")
        fig.tight_layout(); fig.savefig(out/"latency.png", dpi=160); plt.close(fig)
