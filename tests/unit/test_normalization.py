import pytest
from hallubench.datasets.normalization import halu_examples, ragtruth_examples, explicit_mapping_rows

def test_halueval_pair_preserves_original():
    raw = dict(knowledge="Paris is in France", question="Where is Paris?", right_answer="France", hallucinated_answer="Mars")
    a,b = halu_examples(raw, 9, "qa", "commit")
    assert a.group_id == b.group_id and a.label == 'supported' and b.label == 'unsupported'
    assert b.original_label == 'hallucinated' and a.original['knowledge'] == 'Paris is in France'

def test_halueval_rejects_missing():
    with pytest.raises(ValueError): halu_examples({'knowledge': 'a'}, 0, 'qa', 'rev')

def test_ragtruth_offsets_and_original_annotations():
    raw = dict(id="1", source_id="s", response="The city is Mars.", split="test", model="m",
               labels=[dict(start=12,end=16,text="Mars", label_type="Evident Conflict", implicit_true=False)])
    ex = ragtruth_examples(raw, dict(source_id='s',prompt='Where?',source_info='The city is Paris.'), 'hash')[0]
    assert ex.label == 'contradicted' and ex.annotations[0].text == 'Mars' and ex.split == 'test'
    raw['labels'][0]['text'] = 'Jupiter'
    with pytest.raises(ValueError): ragtruth_examples(raw, dict(source_id='s'), 'hash')

def test_unknown_upstream_span_fail_closed():
    raw = dict(id='a',source_id='s',response='Some text',split='test',labels=[dict(start=0,end=4,text='Some',label_type='Not A Type')])
    with pytest.raises(ValueError): ragtruth_examples(raw,dict(source_id='s'),'x')

def test_explicit_map_rejects_unmapped():
    mapping=dict(columns={k:k for k in ('id','answer','label','question','evidence','task','group_id')},labels={'right':'supported'})
    row=dict(id='x',answer='A',label='other',question='Q',evidence='E',task='qa',group_id='g')
    with pytest.raises(ValueError): list(explicit_mapping_rows([row],mapping,'local','rev'))

def test_ragtruth_empty_source_info_is_not_evidence():
    raw=dict(id='1',source_id='s',response='Unknown answer',split='test',labels=[])
    ex=ragtruth_examples(raw, dict(source_id='s',source='MARCO',source_info=''), 'hash')[0]
    assert ex.evidence == ''
