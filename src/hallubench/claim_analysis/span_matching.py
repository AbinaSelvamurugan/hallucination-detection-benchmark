"""Response-grouped span evaluation with explicit character-overlap IoU matching."""
from __future__ import annotations

def iou(a, b):
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return overlap / union if union else 0.0


def span_counts(gold, predicted, threshold=0.5, require_label=False):
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    eligible = [(iou(g, p), i, j) for i, g in enumerate(gold) for j, p in enumerate(predicted)
                if (not require_label or g.label == p.label) and iou(g, p) >= threshold]
    used_g, used_p = set(), set()
    for score, i, j in sorted(eligible, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j)
    return {"tp": len(used_g), "fp": len(predicted)-len(used_p), "fn": len(gold)-len(used_g)}


def span_metrics(items):
    """items: iterable of (gold_spans, predicted_spans) for independent parent responses."""
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for gold, predicted in items:
        c = span_counts(gold, predicted)
        for k in counts:
            counts[k] += c[k]
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    return {**counts, "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "matching_rule": "one-to-one greedy char-IoU >= 0.5; class-agnostic"}
