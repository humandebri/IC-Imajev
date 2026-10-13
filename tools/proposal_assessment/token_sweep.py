"""Shared file and exact numeric utilities for Imajev's retained assessments.

The historical prompt-sweep CLI is retired; this module preserves its utility API.
"""
import hashlib
import json
from pathlib import Path
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[2]
ALIASES = dict(zip(
    ('neuron_minimum_dissolve_delay_to_vote_seconds', 'neuron_minimum_stake_e8s',
     'initial_voting_period_seconds', 'max_dissolve_delay_seconds',
     'max_neuron_age_for_age_bonus', 'wait_for_quiet_deadline_increase_seconds',
     'max_dissolve_delay_bonus_percentage', 'max_age_bonus_percentage', 'reject_cost_e8s'),
    ('min voting lock seconds', 'min stake e8s', 'voting period seconds', 'max lock seconds',
     'max bonus age seconds', 'quiet extension seconds', 'max lock bonus percent',
     'max age bonus percent', 'rejection cost e8s')))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def numeric_change(row):
    field = row['field']
    title = ALIASES[field]
    values = [row.get('previous'), row.get('proposed')]
    if all(type(v) is int for v in values):
        if field.endswith('_e8s'):
            title = title.removesuffix('e8s') + 'tokens'
            with localcontext() as ctx:
                ctx.prec = max(28, max(len(str(abs(v))) for v in values) + 8)
                values = [format(Decimal(v) / 100000000, 'f') for v in values]
        elif title.endswith('seconds') and all(v % 86400 == 0 for v in values):
            title = title.removesuffix('seconds') + 'days'
            values = [v // 86400 for v in values]
    return '{}: {} -> {}'.format(title, *values)
