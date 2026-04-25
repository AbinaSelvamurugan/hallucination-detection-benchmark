"""Transparent evidence token overlap, no truth claims or probability outputs."""
import re
from hallubench.detectors.base import Detector

STOP = frozenset("the a an and or with as in on at to by of for is are was were be had has it this that from their its his her he she they we you not".split())

def informative_tokens(text):
    return {w for w in re.findall(r"[\w]+", (text or "").lower()) if w not in STOP and len(w) > 1}

class HeuristicDetector(Detector):
    name = "lexical_overlap"
    requires_evidence = True

    def __init__(self, threshold=0.45):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self.version = f"1/t={threshold:.3f}"

    def _predict(self, ex):
        evidence = ex.evidence or ex.reference_answer
        terms = informative_tokens(ex.answer)
        ref = informative_tokens(evidence)
        if not terms or not ref:
            return None, {}, "none", {"reason": "Too little lexical content"}
        score = len(terms & ref) / len(terms)
        return ("supported" if score >= self.threshold else "unsupported",
            {"lexical_overlap": float(score)}, "raw", {"threshold": self.threshold})
