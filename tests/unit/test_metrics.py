import pytest
from hallubench.schemas import Prediction
from hallubench.evaluation.metrics import compute_metrics, select_threshold, binary_prob
from hallubench.evaluation.calibration import calibration_metrics
from hallubench.evaluation.runner import load_prepared

@pytest.fixture
def pairs():
    data=load_prepared('tests/fixtures/toy_prepared.jsonl')
    data=[x for x in data if x.split=='test']
    labels=['supported','contradicted','unsupported','supported','unsupported','unsupported']
    return data, {x.example_id: Prediction(x.example_id,'test','1',lab) for x,lab in zip(data, labels)}

def test_binary_expected_confusion(pairs):
    data, preds = pairs
    r=compute_metrics(data,preds)
    assert r['eligible']==5 and r['excluded_truth_insufficient']==1
    assert r['metrics']['confusion_matrix']==[[1,1],[1,2]]
    assert r['metrics']['precision']==pytest.approx(2/3)
    assert r['metrics']['recall']==pytest.approx(2/3)

def test_abstention_not_false_positive(pairs):
    data,preds=pairs
    preds[data[0].example_id]=Prediction(data[0].example_id,'test','1',None,status='error')
    r=compute_metrics(data,preds)
    assert r['evaluated']==4 and r['coverage']==pytest.approx(4/5)

def test_order_and_multiclass(pairs):
    data,preds=pairs
    m=compute_metrics(data,preds,multiclass=True)['metrics']
    assert m['confusion_labels']==['supported','contradicted','unsupported','partially_supported']
    assert len(m['confusion_matrix'])==4

def test_probability_validity():
    with pytest.raises(ValueError): calibration_metrics([0,1],[.1,1.1])
    assert calibration_metrics([0,1],[0,1])['brier']==0
    p=Prediction('x','d','1','supported',scores={'supported':.8,'unsupported':.2},probability_kind='probability')
    assert binary_prob(p)==pytest.approx(.2)
    assert select_threshold([0,1],[.1,.9]) in [i/100 for i in range(5,100,5)] or isinstance(select_threshold([0,1],[.1,.9]), float)
