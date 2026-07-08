from hallubench.evaluation.runner import evaluate
from hallubench.detectors.heuristic import HeuristicDetector
from hallubench.reporting.report import generate_report
from hallubench.storage.results import read_json


def test_full_fixture_run_and_resumption(tmp_path):
    path='tests/fixtures/toy_prepared.jsonl'
    out=tmp_path/'run'
    evaluate(path,[HeuristicDetector()],out)
    assert (out/'predictions_lexical_overlap.jsonl').exists()
    assert read_json(out/'experiment.json')['evaluation_tier']=='fixture_smoke'
    evaluate(path,[HeuristicDetector()],out,resume=True)
    assert read_json(out/'summary.json')['detectors']['lexical_overlap']['n_new_inferences']==0
    assert generate_report(out).exists()
    assert (out/'detector_comparison.png').exists()
