"""Sentence-transformer embeddings, cosine similarity (NOT factuality probabilities)."""
import numpy as np
from hallubench.detectors.base import Detector

class SemanticDetector(Detector):
    name = "semantic_similarity"
    requires_evidence = True

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", revision=None,
                 threshold=0.6, device="cpu", local_files_only=False):
        self.model_id = model
        self.revision = revision
        self.threshold = threshold
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self.version = f"{model}@{revision or 'unpinned'}/t={threshold:.3f}"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, revision=self.revision, device=self.device,
                                             local_files_only=self.local_files_only)

    def _predict(self, ex):
        self._load()
        enc = np.asarray(self._model.encode([ex.evidence or ex.reference_answer, ex.answer],
                            normalize_embeddings=True, convert_to_numpy=True), dtype=float)
        if not np.all(np.isfinite(enc)):
            raise ValueError("Non-finite embedding")
        score = float(np.clip(np.dot(enc[0], enc[1]), -1, 1))
        return ("supported" if score >= self.threshold else "unsupported",
            {"cosine_similarity": score}, "raw",
            {"model_id": self.model_id, "revision": self.revision, "device": self.device,
             "threshold": self.threshold})

    def close(self):
        self._model = None
