from hallubench.claim_analysis.span_matching import span_counts, span_metrics
from hallubench.claim_analysis.segmentation import sentence_spans
from hallubench.schemas import Span

def test_exact_and_partial_span():
    gold=[Span(0,10,'abcdefghij','contradicted')]
    pred=[Span(0,10,'abcdefghij','unsupported')]
    assert span_counts(gold,pred)==dict(tp=1,fp=0,fn=0)
    assert span_counts(gold,pred,require_label=True)==dict(tp=0,fp=1,fn=1)
    assert span_metrics([(gold,[])])['recall']==0

def test_sentence_offsets():
    text='  First fact. Second fact!'
    spans=sentence_spans(text)
    assert spans[0]==(2,13,'First fact.')
    assert text[spans[1][0]:spans[1][1]]=='Second fact!'
