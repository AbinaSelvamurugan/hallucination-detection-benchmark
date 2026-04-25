"""Supervised TF-IDF logistic regression. Training must be explicit."""
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hallubench.detectors.base import Detector
from hallubench.schemas import binary_truth


def features(ex):
    return f"TASK {ex.task} QUESTION {ex.question} EVIDENCE {ex.evidence or ex.reference_answer or ''} ANSWER {ex.answer}"

class TrainedClassifier(Detector):
    name = "tfidf_logreg"
    requires_evidence = False

    def __init__(self, checkpoint=None, threshold=0.5):
        self.threshold = float(threshold)
        self.pipeline = None
        self.version = "untrained"
        if checkpoint:
            payload = joblib.load(checkpoint)  # ONLY load checkpoints from TRUSTED sources
            if payload.get("architecture") != "tfidf_logreg_v1":
                raise ValueError("Unknown checkpoint architecture")
            self.pipeline = payload["pipeline"]
            self.version = payload.get("training_fingerprint", "unknown")
            self.threshold = payload["threshold"]

    def train(self, examples, validation, threshold_candidates=None):
        from hallubench.config import fingerprint
        if not examples or not validation or any(x.split != "train" for x in examples) or any(x.split != "validation" for x in validation):
            raise ValueError("Training requires non-empty strictly train and validation partitions")
        tr = [(x, binary_truth(x)) for x in examples if binary_truth(x) is not None]
        va = [(x, binary_truth(x)) for x in validation if binary_truth(x) is not None]
        if len(set(y for _, y in tr)) != 2 or len(va) == 0:
            raise ValueError("Train requires both classes and labelled validation samples")
        self.pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
                                  ("classifier", LogisticRegression(max_iter=1000, random_state=17))])
        self.pipeline.fit([features(x) for x, _ in tr], [y for _, y in tr])
        probs = self.pipeline.predict_proba([features(x) for x, _ in va])[:, list(self.pipeline.classes_).index(1)]
        labels = np.asarray([y for _, y in va])
        from sklearn.metrics import f1_score
        candidates = threshold_candidates or np.linspace(0.1, 0.9, 17)
        self.threshold = float(max(candidates, key=lambda t: (f1_score(labels, probs >= t, zero_division=0), -abs(t-0.5))))
        self.version = fingerprint({"training_ids": sorted(x.example_id for x, _ in tr),
                                    "validation_ids": sorted(x.example_id for x, _ in va),
                                    "threshold": self.threshold})[:16]
        return {"n_train": len(tr), "n_validation": len(va), "threshold": self.threshold,
                "model_version": self.version, "objective": "validation hallucination F1"}

    def save(self, path):
        if self.pipeline is None:
            raise ValueError("Cannot save untrained model")
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"architecture": "tfidf_logreg_v1", "pipeline": self.pipeline,
                     "training_fingerprint": self.version, "threshold": self.threshold}, path)

    def _predict(self, ex):
        if self.pipeline is None:
            raise RuntimeError("Not trained: specify a trusted checkpoint or run train")
        prob = float(self.pipeline.predict_proba([features(ex)])[0, list(self.pipeline.classes_).index(1)])
        if not np.isfinite(prob):
            raise ValueError("Non-finite score")
        return ("unsupported" if prob >= self.threshold else "supported",
                {"supported": 1-prob, "unsupported": prob}, "probability",
                {"threshold": self.threshold, "trained_checkpoint_version": self.version,
                 "note": "Model probability is not necessarily calibrated"})
