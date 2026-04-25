from hallubench.detectors.heuristic import HeuristicDetector
from hallubench.detectors.semantic_similarity import SemanticDetector
from hallubench.detectors.nli_detector import NLIDetector
from hallubench.detectors.llm_judge import LLMJudge
from hallubench.detectors.classifier import TrainedClassifier

DETECTORS = {"heuristic": HeuristicDetector, "semantic": SemanticDetector, "nli": NLIDetector,
             "judge": LLMJudge, "classifier": TrainedClassifier}

def create(name, **kwargs):
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name}, options: {list(DETECTORS)}")
    return DETECTORS[name](**kwargs)
