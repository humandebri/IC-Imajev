"""Experimental evidence gates before the <=128-token advisory model.

Explicit local policy, not SNS protocol rules or externally verified vote golds.
All recommendations remain conditional on the supplied historical snapshot.
"""
from dataclasses import asdict, dataclass
import json

DAY = 86400


@dataclass(frozen=True)
class CandidatePolicy:
    # An experiment's normative policy parameters, not inferred from GPT labels.
    new_voting_lock_cap_days: int = 365
    minimum_lock_increase_factor: int = 100
    enable_local_reject_gate: bool = False

    def __post_init__(self):
        if (type(self.new_voting_lock_cap_days) is not int or self.new_voting_lock_cap_days < 1 or
            type(self.minimum_lock_increase_factor) is not int or self.minimum_lock_increase_factor < 2):
            raise ValueError('positive cap days and increase factor >=2 required')


def evidence_gate(task, policy=CandidatePolicy()):
    state = json.loads(task['state'])
    action = state.get('action')
    rows = state.get('changes_or_requests')
    if not isinstance(rows, list) or not rows:
        return dict(label='hold', route='evidence_gate', reason='missing changed structured evidence',
                    input_tokens=0, execution_authorization=False)
    result = dict(label=None, route='model', input_tokens=None,
                  execution_authorization=False, snapshot_old_values_chain_verified=False)
    if action == 'ManageNervousSystemParameters' and policy.enable_local_reject_gate:
        matching = [r for r in rows if isinstance(r, dict) and
                    r.get('field') == 'neuron_minimum_dissolve_delay_to_vote_seconds']
        if len(matching) == 1:
            row = matching[0]
            old, new = row.get('previous'), row.get('proposed')
            cap = policy.new_voting_lock_cap_days*DAY
            if (type(old) is int and type(new) is int and old > 0 and old <= cap < new and
                    new >= old*policy.minimum_lock_increase_factor):
                return {**result, 'label': 'reject', 'route': 'candidate_local_policy', 'input_tokens': 0,
                        'reason': 'extreme new voting eligibility lock violates the explicitly configured local policy',
                        'witness': {'field': row['field'], 'old_seconds': old, 'new_seconds': new,
                                    'policy': asdict(policy)},
                        'not_an_sns_protocol_rejection': True, 'production_policy_approved': False}
    for row in rows:
        if not isinstance(row, dict):
            return {**result, 'label': 'hold', 'route': 'evidence_gate', 'input_tokens': 0,
                    'reason': 'malformed changed evidence'}
        unknowns = set(row.get('unknowns', []))
        field = row.get('field')
        if field == 'token_mint' and unknowns & {'supply_before', 'recipient_holdings_before', 'recipient_control'}:
            return {**result, 'label': 'hold', 'route': 'evidence_gate', 'input_tokens': 0,
                    'reason': 'mint concentration/control evidence missing', 'unknowns': sorted(unknowns)}
        if field == 'treasury_transfer' and unknowns & {'treasury_balance_before', 'recipient_control', 'recipient_relationship'}:
            return {**result, 'label': 'hold', 'route': 'evidence_gate', 'input_tokens': 0,
                    'reason': 'treasury capacity/control/recipient relationship evidence missing', 'unknowns': sorted(unknowns)}
        if field == 'generic_call':
            raw, rendered = row.get('payload_sha256'), row.get('rendered_payload_sha256')
            conflict = raw is not None and rendered is not None and raw != rendered
            if conflict or unknowns & {'execution_semantics', 'target_code'}:
                return {**result, 'label': 'hold', 'route': 'evidence_gate', 'input_tokens': 0,
                        'reason': 'executable content missing or payload sources conflict',
                        'payload_conflict': conflict, 'unknowns': sorted(unknowns)}
    return result
