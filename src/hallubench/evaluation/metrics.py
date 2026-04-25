"""Abstention-aware metrics; errors never become predicted positive/negative."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score,
    precision_recall_curve)
from hallubench.schemas import LABELS, binary_truth, binary_prediction


def binary_prob(pred):
    """A valid positive-class score only for a complete supported-versus-hallu probability distribution."""
    if pred.status != "ok" or pred.probability_kind != "probability" or "supported" not in pred.scores:
        return None
    if not set(pred.scores).issubset(set(LABELS)):
        return None
    if abs(sum(pred.scores.values()) - 1) > 1e-3:
        return None
    return float(1 - pred.scores["supported"])


def compute_metrics(examples, predictions, multiclass=False):
    pairs = [(ex, predictions[ex.example_id]) for ex in examples if ex.example_id in predictions]
    if len(pairs) != len(examples):
        missing = sorted(set(x.example_id for x in examples)-set(predictions))
        raise ValueError(f"Missing predictions for examples: {missing[:5]}")
    def truth(ex):
        return ex.label if multiclass and ex.label != "insufficient_evidence" else (None if multiclass else binary_truth(ex))
    def predlabel(pr):
        return pr.label if multiclass and pr.status == "ok" and pr.label != "insufficient_evidence" else (None if multiclass else binary_prediction(pr))
    eligible = [(e, p) for e, p in pairs if truth(e) is not None]
    evaluated = [(e, p) for e, p in eligible if predlabel(p) is not None]
    labels = list(LABELS[:-1]) if multiclass else [0, 1]
    y = [truth(e) for e, p in evaluated]
    pred = [predlabel(p) for e, p in evaluated]
    statuses = {s: sum(p.status == s for e, p in pairs) for s in ("ok", "abstained", "missing_evidence", "error", "unavailable")}
    result = {"total": len(examples), "eligible": len(eligible), "evaluated": len(evaluated),
              "coverage": len(evaluated)/len(eligible) if eligible else 0,
              "statuses": statuses, "excluded_truth_insufficient": len(pairs)-len(eligible),
              "positive_class": "hallucinated" if not multiclass else None,
              "averaging": "support-versus-hallucination binary" if not multiclass else "normalized multiclass"}
    if not y:
        return {**result, "metrics": None, "note": "No eligible non-abstained predictions"}
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    metric = {"accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(macro[2]), "weighted_f1": float(weighted[2]),
        "mcc": float(matthews_corrcoef(y, pred)),
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                     "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "confusion_labels": [str(x) for x in labels],
        "classification_report": classification_report(y, pred, labels=labels, zero_division=0, output_dict=True)}
    if not multiclass:
        metric.update({"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1])})
        scored = [(binary_truth(e), binary_prob(p)) for e, p in evaluated]
        if all(v is not None for _, v in scored) and len(set(y)) == 2:
            from hallubench.evaluation.calibration import calibration_metrics
            scores = np.array([v for _, v in scored], dtype=float)
            metric.update({"roc_auc": float(roc_auc_score(y, scores)),
                           "pr_auc": float(average_precision_score(y, scores)),
                           "calibration": calibration_metrics(np.array(y), scores),
                           "score_note": "Model outputs not independently calibrated; Brier/ECE diagnostic only"})
    return {**result, "metrics": metric}


def select_threshold(validation_true, validation_scores, candidates=None):
    """Maximize validation hallucination F1; never call with test rows."""
    from sklearn.metrics import f1_score
    if not validation_true or len(validation_true) != len(validation_scores):
        raise ValueError("Need aligned labelled validation scores")
    if not set(validation_true).issubset({0, 1}):
        raise ValueError("Targets must be binary")
    candidates = list(candidates if candidates is not None else np.linspace(0.05, 0.95, 19))
    if not candidates:
        raise ValueError("No threshold candidates")
    return float(max(candidates, key=lambda t: (f1_score(validation_true,
        np.array(validation_scores) >= t, zero_division=0), -abs(t-.5))))
