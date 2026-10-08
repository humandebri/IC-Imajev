"""Build a snapshot-only advisory task; never authorizes execution."""
from copy import deepcopy
import json
from .core import CONTRACT, validate_prediction

OPTIONS = CONTRACT['vote_options']
QUESTION = CONTRACT['vote_question']
_OMITTED = {'evidence','review','description','magnitude_ratio','delta','status'}

def make_vote_task(facts):
    unchanged = [r['field'] for r in facts['items'] if r.get('direction')=='unchanged']
    changes = []
    for fact in facts['items']:
        if fact.get('direction')=='unchanged':
            continue
        change = {key:deepcopy(value) for key,value in fact.items() if key not in _OMITTED}
        if change.get('motion_text') in facts['text'].values():
            change.pop('motion_text')
            change['motion_text_same_as_proposer_text'] = True
        changes.append(change)
    state = dict(action=facts['action'],proposer_text=facts['text'],changes_or_requests=changes,
                 unchanged_fields=unchanged,limits=CONTRACT['vote_limits'])
    return dict(protocol_version=1,task_id='approval_recommendation',question=QUESTION,options=list(OPTIONS),
                option_descriptions=OPTIONS,state=json.dumps(state,ensure_ascii=False,separators=(',',':'),allow_nan=False),
                state_projection_version=1,unchanged_values_omitted=unchanged,correctness_verified=False)

def predict_vote(adapter, task):
    try:
        response = validate_prediction(adapter.predict(task),task['options'])
        return dict(response,status='model_prediction',correctness_verified=False,rationale_generated=False,execution_authorization=False)
    except Exception as error:
        return dict(status='unavailable',reason=str(error)[:500],correctness_verified=False,execution_authorization=False)
