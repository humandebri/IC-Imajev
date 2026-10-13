"""Conservative local selection: display-only updates and motions may be skipped."""
import json
POLICY_VERSION = 'important-actions-v1'
EXECUTABLE = {
    'TransferSnsTreasuryFunds':'treasury transfer', 'MintSnsTokens':'token issuance',
    'ExecuteGenericNervousSystemFunction':'custom executable call; include even without topic',
    'UpgradeSnsControlledCanister':'application code upgrade',
    'ManageDappCanisterSettings':'application resource/control settings',
    'RegisterDappCanisters':'governed canister registration', 'DeregisterDappCanisters':'governed canister release',
    'RegisterExtension':'extension registration', 'UpgradeSnsToNextVersion':'SNS framework upgrade',
    'AdvanceSnsTargetVersion':'SNS framework target version',
    'AddGenericNervousSystemFunction':'custom execution function registration',
    'RemoveGenericNervousSystemFunction':'custom execution function removal',
    'SetTopicsForCustomProposals':'custom proposal topic/criticality change',
}
DISPLAY_FIELDS = {'ManageLedgerParameters':{'token_logo','token_name','token_symbol'},
                  'ManageSnsMetadata':{'logo','name','description','url'}}

def select_task(task):
    state = json.loads(task['state'])
    action,rows = state.get('action'),state.get('changes_or_requests')
    selected = True
    if action in EXECUTABLE:
        reason = EXECUTABLE[action]
    elif action == 'Motion':
        selected,reason = False,'non-executing motion excluded by local policy'
    elif action == 'ManageNervousSystemParameters':
        selected = not (rows==[] and bool(state.get('unchanged_fields')))
        reason = ('governance/economic/permission parameters changed or evidence incomplete' if selected
                  else 'only unchanged parameters in projection')
    elif action in DISPLAY_FIELDS and isinstance(rows,list) and rows:
        selected = not all(isinstance(row,dict) and row.get('field') in DISPLAY_FIELDS[action] for row in rows)
        reason = ('ledger/metadata field beyond known display-only fields: review' if selected
                  else 'display/branding-only fields excluded by local policy')
    else:
        reason = 'unknown/incomplete action: review rather than silently skip'
    return dict(policy_version=POLICY_VERSION,action=action,requires_model=selected,reason=reason,
                skipped_is_approval=False,execution_authorization=False)
