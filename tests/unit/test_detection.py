import pytest
from hallubench.detectors.heuristic import HeuristicDetector
from hallubench.detectors.classifier import TrainedClassifier
from hallubench.detectors.nli_detector import NLIDetector
from hallubench.detectors.llm_judge import LLMJudge
from hallubench.schemas import Example
from hallubench.evaluation.runner import load_prepared

def test_heuristic_and_missing_context():
    ex=load_prepared('tests/fixtures/toy_prepared.jsonl')[0]
    p=HeuristicDetector().predict(ex)
    assert p.status=='ok' and p.label in ('supported','unsupported') and p.probability_kind=='raw'
    ex.evidence=None
    assert HeuristicDetector().predict(ex).status=='missing_evidence'

def test_untrained_classifier_explicit_failure():
    ex=load_prepared('tests/fixtures/toy_prepared.jsonl')[0]
    assert TrainedClassifier().predict(ex).status=='error'

def test_train_only_on_dev():
    xs=load_prepared('tests/fixtures/toy_prepared.jsonl')
    c=TrainedClassifier()
    info=c.train([x for x in xs if x.split=='train'],[x for x in xs if x.split=='validation'])
    assert info['n_train']==4 and c.predict(xs[6]).status=='ok'
    with pytest.raises(ValueError): c.train(xs,[x for x in xs if x.split=='test'])

def test_nli_unsafe_config_mapping_fails_without_download():
    n=NLIDetector()
    assert n.requires_evidence
    assert n._model is None

def test_llm_structured_rubric_optional():
    assert 'insufficient_evidence' in __import__('hallubench.detectors.llm_judge', fromlist=['RUBRIC']).RUBRIC
    with pytest.raises(ValueError): LLMJudge('')

def test_detector_exception_is_status_not_label():
    from hallubench.detectors.base import Detector
    class Boom(Detector):
        name='boom'
        def _predict(self, ex):
            raise RuntimeError('offline model')
    ex=load_prepared('tests/fixtures/toy_prepared.jsonl')[0]
    p=Boom().predict(ex)
    assert p.status=='error' and p.label is None and 'offline model' in p.error
