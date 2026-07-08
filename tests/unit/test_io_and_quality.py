import csv
import json
import pytest
from pathlib import Path
from hallubench.config import fingerprint, sha256_file
from hallubench.datasets.hf_loader import load_local, revision_for_local, load_hf
from hallubench.datasets.validation import quality_report
from hallubench.evaluation.runner import load_prepared, evaluate
from hallubench.detectors.heuristic import HeuristicDetector
from hallubench.storage.results import read_jsonl, write_jsonl, read_json
from hallubench.schemas import Prediction
from hallubench.evaluation.metrics import compute_metrics
from hallubench.evaluation.statistics import paired_bootstrap


def test_local_csv_and_jsonl(tmp_path):
    p=tmp_path/'data.csv'
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','label'])
        w.writeheader();w.writerow(dict(id='1',label='supported'))
    assert list(load_local(p)) == [{'id':'1','label':'supported'}]
    j=tmp_path/'data.jsonl'
    write_jsonl(j,[{'id':'2','label':'unsupported'}])
    assert list(load_local(j))[0]['id']=='2'
    assert revision_for_local(j).startswith('sha256:')
    assert sha256_file(j)==revision_for_local(j)[7:]


def test_jsonl_rejects_broken_line(tmp_path):
    p=tmp_path/'broken.jsonl';p.write_text('{not JSON}\n')
    with pytest.raises(ValueError): list(read_jsonl(p))
    with pytest.raises(ValueError): list(load_local('invalid.txt'))


def test_hf_requires_complete_identifiers():
    with pytest.raises(ValueError): list(load_hf('pminervini/HaluEval','','data','revision'))


def test_quality_duplicate_id():
    example=load_prepared('tests/fixtures/toy_prepared.jsonl')[0]
    result=quality_report([example,example],check_leakage=False)
    assert result['duplicates'] and not result['pass']


def test_reject_prediction_id_mismatch(tmp_path):
    from hallubench.detectors.base import Detector
    class Wrong(Detector):
        name='wrong'
        def _predict(self, e):
            return 'supported', {}, 'none', {}
        def predict_batch(self, examples, batch_size=16):
            return [Prediction('mismatched', 'wrong','1','supported') for _ in examples]
    with pytest.raises(ValueError): evaluate('tests/fixtures/toy_prepared.jsonl',[Wrong()],tmp_path)


def test_resume_detects_changed_detector(tmp_path):
    path='tests/fixtures/toy_prepared.jsonl'
    evaluate(path,[HeuristicDetector(.45)],tmp_path)
    with pytest.raises(ValueError): evaluate(path,[HeuristicDetector(.8)],tmp_path,resume=True)


def test_grouped_bootstrap_is_deterministic():
    examples=[x for x in load_prepared('tests/fixtures/toy_prepared.jsonl') if x.split=='test']
    preds={x.example_id:Prediction(x.example_id,'d','1','unsupported') for x in examples}
    x=paired_bootstrap(examples,preds,preds,iterations=20,seed=1)
    y=paired_bootstrap(examples,preds,preds,iterations=20,seed=1)
    assert x == y and x['observed_delta']==0


def test_metric_missing_result_fails_fast():
    examples=[x for x in load_prepared('tests/fixtures/toy_prepared.jsonl') if x.split=='test']
    with pytest.raises(ValueError): compute_metrics(examples,{})
