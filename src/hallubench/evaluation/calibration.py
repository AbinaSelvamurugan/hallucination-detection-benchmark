"""ECE (equal-width bins), reliability data and Brier on valid probabilities only."""
import numpy as np
from sklearn.metrics import brier_score_loss

def calibration_metrics(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.ndim != 1 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Calibration expects paired binary arrays")
    if n_bins < 2 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Requires finite probabilities in [0,1]")
    if not len(y):
        raise ValueError("Empty inputs")
    bins = np.minimum((p*n_bins).astype(int), n_bins-1)
    reliability, ece = [], 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            empirical, predicted = float(y[mask].mean()), float(p[mask].mean())
            ece += mask.mean()*abs(empirical-predicted)
            reliability.append({"bin": b, "count": int(mask.sum()), "observed_frequency": empirical,
                                "mean_predicted": predicted})
    return {"brier": float(brier_score_loss(y, p)), "ece": float(ece),
            "estimator": f"equal-width ECE, {n_bins} bins", "reliability": reliability}
