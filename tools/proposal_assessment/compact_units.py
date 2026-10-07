"""Exact unit-grouped overflow encoding, without IDs or proposal labels."""
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


def integer_words(value):
    """Exact alternatives for round hundreds/thousands, with no rounding."""
    small = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    if value >= 1000 and value % 1000 == 0 and value//1000 < 20:
        return small[value//1000]+' thousand'
    if value >= 100 and value % 100 == 0 and value//100 < 10:
        return small[value//100]+' hundred'
    return str(value)


def encode(task, ratio_words=False):
    state = json.loads(task['state'])
    if state['action'] != 'ManageNervousSystemParameters':
        raise ValueError('parameter changes only')
    rows = state['changes_or_requests']
    if not rows or len({r['field'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique field set required')
    groups = {}; facts = []
    for field, (group, name) in FIELDS.items():
        matching = [r for r in rows if r['field'] == field]
        if not matching:
            continue
        row = matching[0]; values = [row.get('previous'), row.get('proposed')]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('known nonnegative integer old/new required')
        units = group
        if group == 'days':
            scaled = [terminating_decimal(Fraction(v, 86400)) for v in values]
            if any(v is None for v in scaled):
                units = 'seconds'; scaled = list(map(str, values))
        elif group == 'tokens':
            scaled = [quantity(terminating_decimal(Fraction(v, 100000000))).replace(' million','million') for v in values]
        elif group == 'mixed':
            # Each value carries its exact unit; no implicit day rounding.
            scaled = [str(v//86400)+'day' if v % 86400 == 0 else str(v)+'second' for v in values]
        else:
            scaled = list(map(str, values))
        if ratio_words:
            if group == 'bonus%':
                scaled = [integer_words(v) for v in values]
            elif group == 'tokens':
                scaled = [integer_words(v//100000000) if v % 100000000 == 0 and v//100000000 < 1000000 else s
                          for v,s in zip(values,scaled)]
        suffix = ''
        # Prefer the explicit voting-lock multiplier; otherwise use stake.
        priority = ('neuron_minimum_dissolve_delay_to_vote_seconds' if any(r['field']=='neuron_minimum_dissolve_delay_to_vote_seconds' for r in rows)
                    else 'neuron_minimum_stake_e8s')
        if ratio_words and field == priority and min(values)>0 and values[0]!=values[1]:
            ratio = Fraction(values[1],values[0])
            rendered_ratio = terminating_decimal(ratio)
            if rendered_ratio is not None:
                suffix = '(×'+quantity(rendered_ratio).replace(' million','million')+')'
        groups.setdefault(units, []).append(name+' '+'→'.join(scaled)+suffix)
        facts.append(dict(field=field, previous=values[0], proposed=values[1], unit_group=units, rendered_values=scaled))
    if len(facts) != len(rows):
        raise ValueError('unsupported field; no changed field may be omitted')
    text = ' '.join((group+': ' if group != 'mixed' else '')+' '.join(items) for group, items in groups.items())
    return text, facts
