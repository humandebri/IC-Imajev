"""Lossless unit grouping for all changes in a numerical participation task."""
from fractions import Fraction
import json
from .improve_binary import terminating_decimal, quantity

QUESTION = 'All changes: approve lower barriers; reject prohibitive stake/lock'
FIELDS = {
    'initial_voting_period_seconds': ('days', 'vote period'),
    'max_dissolve_delay_seconds': ('days', 'max lock'),
    'max_neuron_age_for_age_bonus': ('days', 'bonus age'),
    'neuron_minimum_dissolve_delay_to_vote_seconds': ('days', 'min voting lock'),
    'max_age_bonus_percentage': ('bonus%', 'age'),
    'max_dissolve_delay_bonus_percentage': ('bonus%', 'lock'),
    'neuron_minimum_stake_e8s': ('tokens', 'stake'),
    'reject_cost_e8s': ('tokens', 'reject cost'),
    'wait_for_quiet_deadline_increase_seconds': ('mixed', 'quiet extension'),
}
_WORDS = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()

def integer_words(value):
    for divisor, limit, unit in ((1000,20,'thousand'), (100,10,'hundred')):
        quotient, remainder = divmod(value, divisor)
        if remainder == 0 and 1 <= quotient < limit:
            return f'{_WORDS[quotient]} {unit}'
    return str(value)

def _values(kind, pair, words):
    if kind == 'mixed':
        return kind, [f'{v//86400}day' if not v%86400 else f'{v}second' for v in pair]
    if kind in ('days', 'tokens'):
        divisor = 86400 if kind == 'days' else 100000000
        decimal = [terminating_decimal(Fraction(v,divisor)) for v in pair]
        if None in decimal:
            return 'seconds', list(map(str,pair))
        if kind == 'days':
            return kind, decimal
        text = []
        for raw, rendered in zip(pair,decimal):
            if words and raw % divisor == 0 and raw//divisor < 1000000:
                text.append(integer_words(raw//divisor))
            else:
                text.append(quantity(rendered).replace(' million','million'))
        return kind, text
    return kind, [integer_words(v) if words else str(v) for v in pair]

def encode(task, ratio_words=False):
    state = json.loads(task['state'])
    if state['action'] != 'ManageNervousSystemParameters':
        raise ValueError('parameter changes only')
    rows = state['changes_or_requests']
    if not isinstance(rows,list) or not rows:
        raise ValueError('nonempty change list required')
    indexed = {}
    for row in rows:
        name = row['field']
        if name in indexed or name not in FIELDS:
            raise ValueError('duplicate or unsupported field; changes cannot be dropped')
        pair = row.get('previous'), row.get('proposed')
        if any(type(v) is not int or v < 0 for v in pair):
            raise ValueError('known nonnegative integer old/new required')
        indexed[name] = pair
    priority = next((name for name in ('neuron_minimum_dissolve_delay_to_vote_seconds',
                                      'neuron_minimum_stake_e8s') if name in indexed), None)
    groups, facts = {}, []
    for field, (kind,label) in FIELDS.items():
        if field not in indexed:
            continue
        pair = indexed[field]
        group, rendered = _values(kind,pair,ratio_words)
        suffix = ''
        if ratio_words and field == priority and min(pair) > 0 and pair[0] != pair[1]:
            ratio = terminating_decimal(Fraction(pair[1],pair[0]))
            if ratio is not None:
                suffix = '(×' + quantity(ratio).replace(' million','million') + ')'
        groups.setdefault(group,[]).append(label + ' ' + '→'.join(rendered) + suffix)
        facts.append(dict(field=field, previous=pair[0], proposed=pair[1], unit_group=group, rendered_values=rendered))
    return ' '.join(('' if unit == 'mixed' else unit + ': ') + ' '.join(parts)
                    for unit,parts in groups.items()), facts
