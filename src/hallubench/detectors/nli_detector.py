"""Bidirectional evidence->claim NLI (claim is hypothesis); neutral != contradiction."""
import numpy as np
from hallubench.detectors.base import Detector

class NLIDetector(Detector):
    name = "nli"
    requires_evidence = True

    def __init__(self, model="cross-encoder/nli-deberta-v3-base", revision=None,
                 device="cpu", local_files_only=False, max_length=512):
        self.model_id, self.revision, self.device = model, revision, device
        self.local_files_only, self.max_length = local_files_only, max_length
        self._model = self._tokenizer = None
        self._label_ids = {}
        self.version = f"{model}@{revision or 'unpinned'}"

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision,
                                                           local_files_only=self.local_files_only)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                           revision=self.revision, local_files_only=self.local_files_only).to(self.device).eval()
            id2label = {int(i): str(v).lower().strip() for i, v in self._model.config.id2label.items()}
            matches = {label: i for i, label in id2label.items() if label in ("entailment", "contradiction", "neutral")}
            if set(matches) != {"entailment", "contradiction", "neutral"} or len(id2label) != 3:
                raise ValueError(f"NLI model label mapping unsafe: {id2label}")
            self._label_ids = matches

    def _predict(self, ex):
        self._load()
        import torch
        reference = ex.evidence or ex.reference_answer
        tokens = self._tokenizer(reference, ex.answer, return_tensors="pt", truncation="only_first",
                                 padding=True, max_length=self.max_length).to(self.device)
        with torch.inference_mode():
            p = torch.softmax(self._model(**tokens).logits[0].float(), dim=-1).cpu().numpy()
        if not np.all(np.isfinite(p)):
            raise ValueError("Non-finite NLI probabilities")
        scores = {name: float(p[i]) for name, i in self._label_ids.items()}
        winner = max(scores, key=scores.get)
        mapped = {"entailment": "supported", "contradiction": "contradicted", "neutral": "unsupported"}[winner]
        return mapped, { {"entailment": "supported", "contradiction": "contradicted", "neutral": "unsupported"}[k]: v
                          for k, v in scores.items() }, "probability", {
            "model_id": self.model_id, "revision": self.revision, "label_ids": self._label_ids,
            "premise": "evidence", "hypothesis": "answer", "max_length": self.max_length,
            "note": "Uncalibrated NLI softmax, neutral maps to lack of entailment, not falsity"}

    def close(self):
        self._model = self._tokenizer = None
