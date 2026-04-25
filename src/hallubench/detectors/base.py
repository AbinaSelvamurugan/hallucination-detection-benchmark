"""Shared status-safe detector contract."""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from hallubench.schemas import Example, Prediction

class Detector(ABC):
    name = "abstract"
    version = "1"
    requires_evidence = False

    @abstractmethod
    def _predict(self, ex: Example) -> tuple[str | None, dict, str, dict]:
        """Return normalized label, scores, probability_kind, metadata."""

    def predict(self, ex: Example) -> Prediction:
        start = time.perf_counter()
        if self.requires_evidence and not (ex.evidence or ex.reference_answer):
            return Prediction(ex.example_id, self.name, self.version, None,
                              status="missing_evidence", error="Reference evidence/answer missing")
        try:
            label, scores, kind, meta = self._predict(ex)
            status = "abstained" if label is None else "ok"
            pred = Prediction(ex.example_id, self.name, self.version, label, scores,
                latency_ms=(time.perf_counter()-start)*1000, status=status,
                metadata=meta, probability_kind=kind)
            pred.validate()
            return pred
        except Exception as exc:
            return Prediction(ex.example_id, self.name, self.version, None,
                latency_ms=(time.perf_counter()-start)*1000,
                status="error", error=f"{type(exc).__name__}: {str(exc)[:500]}")

    def predict_batch(self, examples: list[Example], batch_size: int = 16) -> list[Prediction]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        return [self.predict(ex) for i in range(0, len(examples), batch_size) for ex in examples[i:i+batch_size]]

    def close(self):
        pass
