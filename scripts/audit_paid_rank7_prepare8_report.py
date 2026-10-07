#!/usr/bin/env python3
"""Independent final archive/totals/restoration audit; never infer goal success."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/paid-rank7-prepare8-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=read(D/'summary.json');proof=read(D/'proof/report.json');guards=read(D/'upgrade-guards/verified.json')
    for key in ['complete','baseline_restored','snapshot_deleted','dual_bank','upgrade_receipts_equal']:assert proof[key]
    for key in ['complete','all_32_hidden_verified','all_32_state_verified','saved_candid_replies_verified','paid_core_api_verified','baseline_restored','deterministic_extra_upgrade_guards_verified']:assert summary[key]
    assert not summary['new_dense_f32_direct_capture_verified']
    files=[Path(__file__),D/'summary.json',D/'proof/report.json',D/'upgrade-guards/verified.json']
    for kind in ['workflow_hashes','reference_hashes']:
        for p,h in summary[kind].items():assert sha(ROOT/p)==h,p;files.append(ROOT/p)
    for p,h in read(D/'reporter-entry-hashes.json').items():assert sha(ROOT/p)==h,p;files.append(ROOT/p)
    assert sha(D/'build/full.wasm')==proof['candidate']==summary['module']==guards['module']
    assert guards['complete'] and guards['baseline_restored'] and len(guards['checks'])==4
    before=read(D/'proof/before.json');restored=read(D/'proof/restored.json')
    assert before['module']==restored['module']=='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
    assert all(restored[k]for k in ['cache_equal','pack_equal','snapshot_deleted'])
    helper=ROOT/'artifacts/paid-update-v1/tools/args'
    def decode(kind,p):return json.loads(subprocess.check_output([str(helper),'decode',kind,str(p)],text=True))
    count=0
    for base in [D/'proof',D/'upgrade-guards']:
        for p in sorted(base.rglob('*.json')):
            call=read(p)
            if not isinstance(call,dict)or not {'kind','reply_path','args_path','result'}<=call.keys():continue
            reply=ROOT/call['reply_path'];argument=ROOT/call['args_path'];raw=bytes.fromhex(reply.read_text().strip().removeprefix('0x'))
            assert len(raw)==call['reply_bytes'] and argument.stat().st_size==call['request_bytes']
            if call['relay']:
                forward=decode('forward',reply);assert forward==call['forward']
                if 'Ok'in forward['response']:
                    inner=reply.with_name(reply.name.replace('.reply.hex','.inner.hex'))
                    assert bytes.fromhex(inner.read_text().strip().removeprefix('0x'))==bytes(forward['response']['Ok'])
                    actual=decode(call['kind'],inner)
                else:actual={'transport_error':forward['response']['Err']}
            else:actual=decode(call['kind'],reply)
            assert actual==call['result'];count+=1
    assert count>30
    prior=read(ROOT/'artifacts/paid-common-raw-hybrid-retry-v3/proof/report.json');totals={};comparisons=[]
    assert {v['case']for v in proof['results']}=={'617','620','653'}
    for item in proof['results']:
        name=item['case'];workers=item['row']['result']['Ok']['workers'];total=sum(w['instructions']for w in workers)
        old=next(v for v in prior['results']if v['case']==name)
        previous=sum(w['instructions']for w in old['row']['result']['Ok']['workers'])
        row=next(v for v in summary['cases']if v['case']==name)
        assert row['total_handler_instructions']==total and row['prior_total_handler_instructions']==previous
        assert item['request']==old['request'] and item['quote']==old['quote']
        assert item['debug']['hidden_hashes']==old['debug']['hidden_hashes'] and item['debug']['state_hashes']==old['debug']['state_hashes']
        for key in ['raw_logits','probabilities','unknown_probability']:
            # Bit comparison, independently materialized F32 from raw decoded Candid values.
            import numpy as np
            a=np.asarray(item['row']['result']['Ok']['decision'][key],dtype='<f4').tobytes()
            b=np.asarray(old['row']['result']['Ok']['decision'][key],dtype='<f4').tobytes();assert a==b
        assert workers and all(w['instructions']<40_000_000_000 and w['heap_pages']*65536<2**32 for w in workers)
        totals[name]=total;comparisons.append(dict(case=name,total=total,previous=previous,saved=previous-total,target_met=total<=100_000_000_000))
    goal=all(v['target_met']for v in comparisons)
    assert summary['all_targets_met']==goal
    with zipfile.ZipFile(D/'frozen-paid-proof.zip')as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip()is None
        assert z.read(str((D/'summary.json').relative_to(ROOT)))==(D/'summary.json').read_bytes()
        for p,h in summary['workflow_hashes'].items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
    live=json.loads(subprocess.check_output(['icp','canister','status','4caro-hl777-77775-aaaba-cai','--network','local','--identity','imajev-local','--json'],text=True))
    assert live['status']=='Running' and live['module_hash'].removeprefix('0x')==before['module']
    result=dict(complete=True,module=summary['module'],raw_candid_calls_redecoded=count,worker_totals=totals,comparisons=comparisons,all3_improve=all(v['saved']>0 for v in comparisons),all_32_hidden_report_and_reference_hashes_verified=True,zip_unique_latest_summary_verified=True,workflow_hashes_verified=True,baseline_restored=True,temporary_snapshot_deleted=True,deterministic_extra_upgrade_guards_verified=True,all_targets_met=goal,latest_dense_direct_capture_complete=False,archive_sha256=sha(D/'frozen-paid-proof.zip'),audit_script_sha256=sha(Path(__file__)))
    (D/'post-report-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
