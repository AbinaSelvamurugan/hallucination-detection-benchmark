import pytest
from hallubench.datasets.splitting import group_split, cross_split_collisions
from hallubench.schemas import Example

def ex(i, group, split='unspecified', question=None):
    return Example(example_id=str(i),question=question or f'Question {i}?',evidence='Earth is blue.',reference_answer=None,
                   answer=f'Response {i}',original_label='supported',label='supported',
                   dataset='toy',revision='1',task='qa',group_id=group,split=split)

def test_same_source_and_question_must_stay_together():
    data = [ex(1,'x'),ex(2,'x'),ex(3,'y',question='Question 1?'),ex(4,'z')]
    group_split(data,seed=2)
    assert data[0].split == data[1].split == data[2].split

def test_existing_cross_split_leakage_raises():
    with pytest.raises(ValueError): group_split([ex(1,'x','train'),ex(2,'x','test')])

def test_exact_dup_answer_detected_across_splits():
    a,b=ex(1,'a','train'),ex(2,'b','test')
    a.answer=b.answer='The same distinctive response is included in two sources.'
    assert cross_split_collisions([a,b])

def test_generic_summarization_prompts_not_forced_to_one_group():
    a,b=ex(1,'a'),ex(2,'b')
    a.question=b.question='Summarize the evidence.'
    group_split([a,b],seed=1)
    assert all(x.split in ('train','validation','test') for x in (a,b))

def test_official_test_preserved_and_train_can_be_carved():
    data=[ex(i,str(i),'train') for i in range(100)] + [ex(101,'reserved','test')]
    group_split(data,seed=17)
    assert data[-1].split=='test'
    assert 0 < sum(x.split=='validation' for x in data) < 100

def test_matching_long_source_documents_are_grouped():
    a,b=ex(1,'a'),ex(2,'b')
    a.evidence=b.evidence='This is the exact same distinctive longer source document for two separately indexed records in different upstream datasets.'
    group_split([a,b])
    assert a.split == b.split
