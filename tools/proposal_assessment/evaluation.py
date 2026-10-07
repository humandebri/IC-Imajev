"""Offline scoring against explicit human labels bound to frozen snapshot hashes."""
TASK_IDS = {'rationale', 'affected_users', 'mitigation'}
LABELS = {'present', 'absent', 'unclear'}


def evaluate(comparison, labels):
    """Unavailable predictions stay in coverage and the overall denominator."""
    if labels.get('schema_version') != 1 or not isinstance(labels.get('snapshots'), dict) or not labels['snapshots']:
        raise ValueError('labels require schema_version=1 and nonempty snapshots')
    snapshots = {p['snapshot_sha256']: p for p in comparison['proposals']}
    gold = labels['snapshots']
    for snapshot, tasks in gold.items():
        if snapshot not in snapshots:
            raise ValueError('label snapshot hash does not match an input')
        if (not isinstance(tasks, dict) or not tasks or set(tasks) - TASK_IDS
                or any(not isinstance(v, str) or v not in LABELS for v in tasks.values())):
            raise ValueError('invalid task labels')
    expected = sum(len(tasks) for tasks in gold.values())
    counters = {}
    for snapshot, tasks in gold.items():
        for trial in snapshots[snapshot]['model_assessments']:
            name = trial['model']['name']
            counter = counters.setdefault(name, {'expected': expected, 'predicted': 0, 'correct': 0,
                                                  'errors': [], 'unavailable': []})
            rows = {r['task_id']: r for r in trial['advisory']}
            for task, label in tasks.items():
                row = rows.get(task, {})
                if row.get('status') != 'model_prediction':
                    counter['unavailable'].append({'snapshot': snapshot, 'task_id': task})
                    continue
                counter['predicted'] += 1
                if row['label'] == label:
                    counter['correct'] += 1
                else:
                    counter['errors'].append({'snapshot': snapshot, 'task_id': task,
                                              'expected': label, 'predicted': row['label']})
    for counter in counters.values():
        counter['coverage'] = counter['predicted'] / expected
        counter['accuracy_on_predictions'] = (counter['correct'] / counter['predicted']
                                               if counter['predicted'] else None)
        counter['correct_over_all_labeled_tasks'] = counter['correct'] / expected
    return {'scope': 'explicitly labeled text tasks only; not proposal safety accuracy', 'models': counters}
