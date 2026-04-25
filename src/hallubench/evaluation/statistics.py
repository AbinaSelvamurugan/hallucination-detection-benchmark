"""Grouped paired bootstrap (group ids are the resampling units)."""
from collections import defaultdict
import numpy as np
from hallubench.schemas import binary_truth, binary_prediction
from sklearn.metrics import f1_score


def paired_bootstrap(examples, predictions_a, predictions_b, iterations=1000, seed=17):
    grouped = defaultdict(list)
    for ex in examples:
        if ex.example_id not in predictions_a or ex.example_id not in predictions_b:
            continue
        y, a, b = binary_truth(ex), binary_prediction(predictions_a[ex.example_id]), binary_prediction(predictions_b[ex.example_id])
        if None not in (y, a, b):
            grouped[ex.group_id].append((y, a, b))
    if not grouped:
        return {"status": "not_estimated", "reason": "No paired labelled non-abstained examples"}
    rng = np.random.default_rng(seed)
    groups = list(grouped)
    diffs = []
    for _ in range(iterations):
        sampled = [point for i in rng.integers(0, len(groups), len(groups)) for point in grouped[groups[i]]]
        y, a, b = zip(*sampled)
        diffs.append(f1_score(y, a, zero_division=0)-f1_score(y, b, zero_division=0))
    observed = [p for g in groups for p in grouped[g]]
    y, a, b = zip(*observed)
    return {"status": "estimated", "unit": "source/parent group", "n_groups": len(groups),
            "n_pairs": len(observed), "iterations": iterations, "seed": seed,
            "metric": "hallucination F1 delta A-B",
            "observed_delta": float(f1_score(y, a, zero_division=0)-f1_score(y, b, zero_division=0)),
            "ci95_percentile": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
            "warning": "Exploratory; no multiplicity correction, CI unstable for few source groups"}
