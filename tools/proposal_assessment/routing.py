"""Select economically or operationally consequential proposals without a model.

This is a local review policy, not the SNS protocol's Critical classification.
Skipping is not approval, rejection, or authorization to vote.
"""
import json

POLICY_VERSION = 'important-actions-v1'
EXECUTABLE = {
    'TransferSnsTreasuryFunds': 'treasury transfer',
    'MintSnsTokens': 'token issuance',
    'ExecuteGenericNervousSystemFunction': 'custom executable call; include even without topic',
    'UpgradeSnsControlledCanister': 'application code upgrade',
    'ManageDappCanisterSettings': 'application resource/control settings',
    'RegisterDappCanisters': 'governed canister registration',
    'DeregisterDappCanisters': 'governed canister release',
    'RegisterExtension': 'extension registration',
    'UpgradeSnsToNextVersion': 'SNS framework upgrade',
    'AdvanceSnsTargetVersion': 'SNS framework target version',
    'AddGenericNervousSystemFunction': 'custom execution function registration',
    'RemoveGenericNervousSystemFunction': 'custom execution function removal',
    'SetTopicsForCustomProposals': 'custom proposal topic/criticality change',
}
DISPLAY_FIELDS = {
    'ManageLedgerParameters': {'token_logo', 'token_name', 'token_symbol'},
    'ManageSnsMetadata': {'logo', 'name', 'description', 'url'},
}


def select_task(task):
    state = json.loads(task['state'])
    action = state.get('action')
    rows = state.get('changes_or_requests')
    reason, selected = 'unknown/incomplete action: review rather than silently skip', True
    if action == 'Motion':
        reason, selected = 'non-executing motion excluded by local policy', False
    elif action in EXECUTABLE:
        reason = EXECUTABLE[action]
    elif action == 'ManageNervousSystemParameters':
        reason = 'governance/economic/permission parameters changed or evidence incomplete'
        if rows == []:
            # A known list of unchanged fields proves this projection found no change.
            if state.get('unchanged_fields'):
                reason, selected = 'only unchanged parameters in projection', False
    elif action in DISPLAY_FIELDS and isinstance(rows, list) and rows:
        if all(isinstance(row, dict) and row.get('field') in DISPLAY_FIELDS[action] for row in rows):
            reason, selected = 'display/branding-only fields excluded by local policy', False
        else:
            reason = 'ledger/metadata field beyond known display-only fields: review'
    return {'policy_version': POLICY_VERSION, 'action': action,
            'requires_model': selected, 'reason': reason,
            'skipped_is_approval': False, 'execution_authorization': False}
