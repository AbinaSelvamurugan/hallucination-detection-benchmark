"""Transparent error lists, rule-based *suggestions* only; no asserted root causes."""
import re
from hallubench.schemas import binary_truth, binary_prediction

def suggest_category(ex):
    text = ex.answer
    if re.search(r"\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion)\b", text):
        return "date_or_quantity_candidate"
    if re.search(r"\b(?:never|not|no|none|without)\b", text.lower()):
        return "negation_candidate"
    if re.search(r"\b(?:because|caused|therefore|led to|resulted in)\b", text.lower()):
        return "causal_candidate"
    return "uncategorized_requires_human_review"

def error_rows(examples, predictions):
    result = []
    for ex in examples:
        p = predictions.get(ex.example_id)
        if not p:
            continue
        truth, predicted = binary_truth(ex), binary_prediction(p)
        if truth is None or predicted is None or truth == predicted:
            continue
        result.append({"example_id": ex.example_id, "dataset": ex.dataset, "task": ex.task,
            "generator": ex.generator, "group_id": ex.group_id, "truth": truth, "prediction": predicted,
            "error_type": "false_negative" if truth == 1 else "false_positive",
            "suggested_category": suggest_category(ex), "category_provenance": "unvalidated regex suggestion",
            "answer_excerpt": ex.answer[:220], "evidence_excerpt": (ex.evidence or "")[:220]})
    return result
