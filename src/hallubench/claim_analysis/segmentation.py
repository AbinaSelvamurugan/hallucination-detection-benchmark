"""Deterministic sentence boundaries, a lossy claim proxy; never claim atomic extraction."""
from __future__ import annotations
import re
from hallubench.schemas import Span

def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    result = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text):
        a, b = m.span()
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        if a < b:
            result.append((a, b, text[a:b]))
    return result


def predicted_error_spans(detector, ex):
    """Apply detector to each sentence; NOT a validated atomic-claim evaluator."""
    from dataclasses import replace
    error_spans, records = [], []
    for n, (a, b, text) in enumerate(sentence_spans(ex.answer)):
        pred = detector.predict(replace(ex, example_id=f"{ex.example_id}:sentence:{n}", answer=text,
                                         annotations=[]))
        records.append({"parent_id": ex.example_id, "claim_id": f"{ex.example_id}:sentence:{n}",
            "start": a, "end": b, "text": text, "label": pred.label,
            "status": pred.status, "provenance": "heuristic sentence segmentation, not gold claims"})
        if pred.status == "ok" and pred.label in ("contradicted", "unsupported", "partially_supported"):
            error_spans.append(Span(a, b, text, pred.label, provenance="predicted sentence proxy"))
    return error_spans, records
