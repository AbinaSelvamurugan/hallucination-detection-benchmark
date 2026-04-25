"""Strict canonical data and prediction schemas."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import math

LABELS = ("supported", "contradicted", "unsupported", "partially_supported", "insufficient_evidence")
HALLUCINATION = ("contradicted", "unsupported", "partially_supported")

@dataclass
class Span:
    start: int
    end: int
    text: str
    label: str
    provenance: str = "upstream"
    original_label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def validate(self, text: str) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Invalid span label: {self.label}")
        if not (isinstance(self.start, int) and isinstance(self.end, int) and 0 <= self.start < self.end <= len(text)):
            raise ValueError(f"Span offsets out of bounds: {self.start}:{self.end}")
        if text[self.start:self.end] != self.text:
            raise ValueError("Span text does not match response at offsets")

@dataclass
class Example:
    example_id: str
    question: str
    evidence: str | None
    reference_answer: str | None
    answer: str
    original_label: str
    label: str
    dataset: str
    revision: str
    task: str
    group_id: str
    split: str = "unspecified"
    source_id: str | None = None
    generator: str | None = None
    annotations: list[Span] = field(default_factory=list)
    provenance: str = "upstream"
    confidence: float | None = None
    original: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Example":
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        fields["annotations"] = [Span.from_dict(s) for s in fields.get("annotations", [])]
        return cls(**fields)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for k in ("example_id", "answer", "original_label", "dataset", "revision", "task", "group_id"):
            if not isinstance(getattr(self, k), str) or not getattr(self, k).strip():
                raise ValueError(f"{self.example_id}: missing {k}")
        if self.label not in LABELS:
            raise ValueError(f"{self.example_id}: unknown label {self.label!r}")
        if self.split not in ("train", "validation", "test", "unspecified"):
            raise ValueError("Invalid split")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Invalid confidence")
        for span in self.annotations:
            span.validate(self.answer)

@dataclass
class Prediction:
    example_id: str
    detector: str
    version: str
    label: str | None
    scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    status: str = "ok"  # ok | abstained | missing_evidence | error | unavailable
    error: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    probability_kind: str = "none"  # none | probability | raw

    def validate(self) -> None:
        if self.status not in ("ok", "abstained", "missing_evidence", "error", "unavailable"):
            raise ValueError("Invalid processing status")
        if self.status == "ok" and self.label not in LABELS:
            raise ValueError("Successful predictions must use normalized labels")
        if self.status != "ok" and self.label is not None:
            raise ValueError("Failed/abstained outputs must have null labels")
        if self.probability_kind not in ("none", "probability", "raw"):
            raise ValueError("Invalid score type")
        for key, val in self.scores.items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Invalid score {key}: {val}")
            if self.probability_kind == "probability" and not (0 <= val <= 1):
                raise ValueError("Probability out of bounds")
        if self.probability_kind == "probability" and self.scores and abs(sum(self.scores.values()) - 1) > 1e-3:
            raise ValueError("Class probabilities must sum to one")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

def binary_truth(example: Example) -> int | None:
    """Coarse evidence-grounded benchmark target, POSITIVE = detected hallucination."""
    if example.label == "supported":
        return 0
    if example.label in HALLUCINATION:
        return 1
    return None

def binary_prediction(pred: Prediction) -> int | None:
    if pred.status != "ok":
        return None
    if pred.label == "supported":
        return 0
    if pred.label in HALLUCINATION:
        return 1
    return None
