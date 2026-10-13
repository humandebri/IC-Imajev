"""Score explicit labels tied to snapshot hashes; missing predictions reduce coverage."""

def evaluate(comparison, labels):
    gold = labels.get('snapshots')
    if labels.get('schema_version')!=1 or not isinstance(gold,dict) or not gold:
        raise ValueError('labels require schema_version=1 and nonempty snapshots')
    inputs = {p['snapshot_sha256']:p for p in comparison['proposals']}
    for digest,tasks in gold.items():
        if digest not in inputs:
            raise ValueError('label snapshot hash does not match an input')
        if not isinstance(tasks,dict) or not tasks or not set(tasks)<={'rationale','affected_users','mitigation'} or any(type(label) is not str or label not in {'present','absent','unclear'} for label in tasks.values()):
            raise ValueError('invalid task labels')
    total = sum(map(len,gold.values()))
    counters = {}
    for digest,tasks in gold.items():
        for trial in inputs[digest]['model_assessments']:
            name = trial['model']['name']
            counter = counters.setdefault(name,dict(expected=total,predicted=0,correct=0,errors=[],unavailable=[]))
            rows = {row['task_id']:row for row in trial['advisory']}
            for task,expected in tasks.items():
                prediction = rows.get(task,{})
                if prediction.get('status')!='model_prediction':
                    counter['unavailable'].append(dict(snapshot=digest,task_id=task))
                else:
                    counter['predicted'] += 1
                    if prediction['label']==expected:
                        counter['correct'] += 1
                    else:
                        counter['errors'].append(dict(snapshot=digest,task_id=task,expected=expected,predicted=prediction['label']))
    for counter in counters.values():
        counter.update(coverage=counter['predicted']/total,
                       accuracy_on_predictions=counter['correct']/counter['predicted'] if counter['predicted'] else None,
                       correct_over_all_labeled_tasks=counter['correct']/total)
    return dict(scope='explicitly labeled text tasks only; not proposal safety accuracy',models=counters)
