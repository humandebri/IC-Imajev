#!/usr/bin/env python3
"""One-shot authorized mainnet funding, including ICP ledger fee in the 6.5 ICP cap."""
import argparse,json,pathlib,subprocess,time
from decimal import Decimal
from mainnet_cli import MainnetCLI,ROOT,TARGET,IDENTITY,atomic_json
CAP_E8S=650_000_000
TRANSFER_FEE_E8S=10_000
MINT_ICP='6.4999'
CYCLES_WITHDRAW_FEE=100_000_000

def cli(args):
    p=subprocess.run(['icp',*args,'--network','ic','--identity',IDENTITY],text=True,capture_output=True,cwd=ROOT,timeout=300)
    if p.returncode:raise RuntimeError(f'CLI failed; outcome may be unknown, never automatically resubmit financial calls: {p.stderr}')
    return p.stdout.strip()
def cycles(s):return int(s.replace('_','').removesuffix(' cycles'))
def e8s(s):return int(Decimal(s.removesuffix(' ICP'))*100_000_000)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--directory',default='artifacts/mainnet-deploy-20261007');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
    plan={'max_authorized_ICP_e8s':CAP_E8S,'mint_ICP':MINT_ICP,'ICP_transfer_fee_e8s':TRANSFER_FEE_E8S,'planned_ICP_debit_e8s':int(Decimal(MINT_ICP)*100_000_000)+TRANSFER_FEE_E8S,'target':TARGET,'identity':IDENTITY,'cycles_withdraw_fee':CYCLES_WITHDRAW_FEE,'wasm_memory_limit_only':4294967296,'no_snapshot':True,'do_not_change_controllers_or_freezing_period':True,'do_not_enable_paid_service':True}
    assert plan['planned_ICP_debit_e8s']<=CAP_E8S
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    c=MainnetCLI(d/'funding-checks');s=c.status()
    beforeICP=json.loads(cli(['token','balance','--json']))
    beforeCycles=json.loads(cli(['cycles','balance','--json']))
    # Financial intent files prohibit replay after an unknown transfer outcome.
    mint=d/'mint-6.4999.receipt.json';intent=d/'mint-6.4999.intent.json'
    if not mint.exists() and e8s(beforeICP['balance'])<CAP_E8S:raise RuntimeError('Insufficient ICP for cap-inclusive conversion')
    if intent.exists() and json.loads(intent.read_text())['plan']!=plan:raise RuntimeError('Financial intent differs from current authorization')
    if mint.exists():result=json.loads(mint.read_text())
    else:
        if intent.exists():raise RuntimeError('Unresolved financial mint intent; inspect ledger before any retry')
        atomic_json(intent,{'plan':plan,'before_ICP':beforeICP,'before_cycles':beforeCycles,'before_canister':s,'started_unix':time.time()})
        result=json.loads(cli(['cycles','mint','--icp',MINT_ICP,'--json']));atomic_json(mint,result)
    amount=cycles(result['deposited'])-CYCLES_WITHDRAW_FEE
    if amount<=0:raise RuntimeError('Invalid minted amount')
    topup=d/'model-topup.receipt.json';ti=d/'model-topup.intent.json'
    if not topup.exists():
        if ti.exists():raise RuntimeError('Unresolved financial topup intent; inspect ledger before retry')
        atomic_json(ti,{'target':TARGET,'amount_cycles':amount,'started_unix':time.time()})
        output=cli(['canister','top-up',TARGET,'--amount',str(amount)])
        atomic_json(topup,{'target':TARGET,'amount_cycles':amount,'CLI_success':True,'stdout':output})
    afterICP=json.loads(cli(['token','balance','--json']));afterCycles=json.loads(cli(['cycles','balance','--json']));after=c.status()
    initial=json.loads(intent.read_text())['before_ICP']['balance']
    report={'authorization':plan,'mint':result,'topup':json.loads(topup.read_text()),'before_ICP':initial,'after_ICP':afterICP['balance'],'observed_ICP_debit_e8s':e8s(initial)-e8s(afterICP['balance']),'after_identity_cycles':afterCycles['balance'],'after_canister':after,'note':'Balance difference may include external deposits; cap is enforced by fixed requested mint + known ICP transfer fee. No automatic further mint/topup.'}
    atomic_json(d/'funding-verified.json',report)
    print(json.dumps({'stage':'funded','after_ICP':afterICP['balance'],'target_cycles':after['cycles'],'amount_deposited':amount}),flush=True)
if __name__=='__main__':main()
