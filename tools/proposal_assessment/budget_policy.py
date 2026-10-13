"""Snapshot evidence gate; local rejection policy is disabled by default."""
from dataclasses import dataclass, asdict
import json
DAY = 86400

@dataclass(frozen=True)
class CandidatePolicy:
    new_voting_lock_cap_days: int = 365
    minimum_lock_increase_factor: int = 100
    enable_local_reject_gate: bool = False
    def __post_init__(self):
        for value,minimum in ((self.new_voting_lock_cap_days,1),(self.minimum_lock_increase_factor,2)):
            if type(value) is not int or value < minimum:
                raise ValueError('positive cap days and increase factor >=2 required')

_REQUIRED = {
    'token_mint':({'supply_before','recipient_holdings_before','recipient_control'},'mint concentration/control evidence missing'),
    'treasury_transfer':({'treasury_balance_before','recipient_control','recipient_relationship'},'treasury capacity/control/recipient relationship evidence missing'),
    'generic_call':({'execution_semantics','target_code'},'executable content missing or payload sources conflict'),
}

def evidence_gate(task, policy=CandidatePolicy()):
    state = json.loads(task['state'])
    rows = state.get('changes_or_requests')
    if not isinstance(rows,list) or not rows:
        return dict(label='hold',route='evidence_gate',reason='missing changed structured evidence',input_tokens=0,execution_authorization=False)
    base = dict(label=None,route='model',input_tokens=None,execution_authorization=False,snapshot_old_values_chain_verified=False)
    if policy.enable_local_reject_gate and state.get('action')=='ManageNervousSystemParameters':
        lock = [r for r in rows if isinstance(r,dict) and r.get('field')=='neuron_minimum_dissolve_delay_to_vote_seconds']
        if len(lock)==1:
            previous,proposed = lock[0].get('previous'),lock[0].get('proposed')
            cap = policy.new_voting_lock_cap_days * DAY
            if type(previous) is int and type(proposed) is int and 0<previous<=cap<proposed and proposed>=previous*policy.minimum_lock_increase_factor:
                return dict(base,label='reject',route='candidate_local_policy',input_tokens=0,
                            reason='extreme new voting eligibility lock violates the explicitly configured local policy',
                            witness=dict(field=lock[0]['field'],old_seconds=previous,new_seconds=proposed,policy=asdict(policy)),
                            not_an_sns_protocol_rejection=True,production_policy_approved=False)
    for row in rows:
        if not isinstance(row,dict):
            return dict(base,label='hold',route='evidence_gate',input_tokens=0,reason='malformed changed evidence')
        field = row.get('field')
        if field not in _REQUIRED:
            continue
        unknowns = set(row.get('unknowns',[]))
        required,reason = _REQUIRED[field]
        conflict = field=='generic_call' and row.get('payload_sha256') is not None and row.get('rendered_payload_sha256') is not None and row['payload_sha256']!=row['rendered_payload_sha256']
        if unknowns & required or conflict:
            result = dict(base,label='hold',route='evidence_gate',input_tokens=0,reason=reason,unknowns=sorted(unknowns))
            if field=='generic_call':
                result['payload_conflict'] = conflict
            return result
    return base
