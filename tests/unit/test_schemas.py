import pytest
from hallubench.schemas import Example, Span, Prediction, binary_truth

def sample(**kw):
    d = dict(example_id="x", question="q", evidence="evidence", reference_answer=None,
             answer="A word.", original_label="yes", label="supported", dataset="Toy",
             revision="rev", task="qa", group_id="g")
    d.update(kw)
    return Example(**d)

def test_valid_schema_roundtrip():
    ex = sample(annotations=[Span(0, 1, "A", "supported")])
    ex.validate()
    assert Example.from_dict(ex.to_dict()).annotations[0].text == "A"

def test_invalid_unknown_labels_and_spans():
    with pytest.raises(ValueError): sample(label="truthy").validate()
    with pytest.raises(ValueError): sample(annotations=[Span(0,2,"X", "supported")]).validate()

def test_binary_mapping_distinguishes_insufficient():
    assert binary_truth(sample(label="contradicted")) == 1
    assert binary_truth(sample(label="unsupported")) == 1
    assert binary_truth(sample(label="partially_supported")) == 1
    assert binary_truth(sample(label="supported")) == 0
    assert binary_truth(sample(label="insufficient_evidence")) is None

def test_failed_prediction_never_counts_as_label():
    with pytest.raises(ValueError): Prediction("x","d","v","supported",status="error").validate()
    with pytest.raises(ValueError): Prediction("x","d","v","supported",scores={"supported": .3},probability_kind="probability").validate()
