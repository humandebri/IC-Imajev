#!/usr/bin/env python3
"""Independently verify saved 32-query results and all exported numerical states."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proof',required=True);args=ap.parse_args();d=ROOT/args.proof
    sources=json.loads((d/'sources.json').read_text())
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    restored=json.loads((d/'restored.json').read_text());assert restored['restored'] and restored['cache_equal'] and restored['pack_equal']
    outcomes=[]
    for proposal in ['653','620']:
        p=d/proposal/'report.json'
        if not p.exists():continue
        r=json.loads(p.read_text());old=ROOT/f'artifacts/boomdao-current-v1/{proposal}-r1';baseline=json.loads((old/'report.json').read_text())
        assert r['query_count']==r['executed_query_count']==len(r['queries'])==32 and r['replayed_queries']==0
        assert not (p.parent/'fallback.json').exists() and not (p.parent/'queries/failures.jsonl').exists()
        assert r['input_hash']==baseline['input_hash'] and r['model']==baseline['model'] and r['pack_hash']==baseline['pack_hash']
        assert r['wasm_sha256']==sha(ROOT/'artifacts/query32-v1/full-build/full.wasm')
        numerical=[]
        for file in sorted((p.parent/'queries').glob('layer-*.npy')):
            ref=old/'queries'/file.name
            assert ref.exists() and np.array_equal(np.load(file).view('<u4'),np.load(ref).view('<u4')),file
            numerical.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),reference=str(ref.relative_to(ROOT))))
        assert len(numerical)==31
        states=[]
        for file in sorted((p.parent/'queries/states').glob('*.npz')):
            ref=old/'queries/states'/file.name;a=np.load(file);b=np.load(ref)
            assert set(a.files)==set(b.files),file
            for key in a.files:assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),(file,key)
            states.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),reference=str(ref.relative_to(ROOT))))
        assert len(states)==32
        final=p.parent/'final-hidden.npy';assert np.load(final).tobytes()==np.load(old/'final-hidden.npy').tobytes()
        a=r['decision_query']['ok']['decision'];b=baseline['decision_query']['ok']['decision']
        assert {k:v for k,v in a.items() if k!='instructions'}=={k:v for k,v in b.items() if k!='instructions'}
        for q in r['queries']:
            index=q['index'];folder=p.parent/'queries';raw=folder/f'{index:06d}.response.bin';header,v=decode(raw.read_bytes())
            assert header['step']==index+1 and header['model']==r['model'] and header['pack_hash']==r['pack_hash']
            assert all(sha(folder/name)==h for name,h in q.get('outputs_sha256',{}).items())
            assert 0<q['ok']['instructions']<5_000_000_000 and q['ok']['heap_pages']*65536<2**32
            assert q['ok']['request_bytes']<2_000_000 and q['ok']['reply_bytes']<2_000_000
        assert sum(q['ok']['instructions'] for q in r['queries'])==r['total_instructions']
        assert sum(q['ok']['request_bytes']+q['ok']['reply_bytes'] for q in r['queries'])==r['total_candid_bytes']
        outcomes.append(dict(proposal=proposal,query_count=32,baseline_query_count=39,total_instructions=r['total_instructions'],instruction_reduction_percent=100*(1-r['total_instructions']/baseline['total_instructions']),candid_bytes=r['total_candid_bytes'],candid_reduction_percent=100*(1-r['total_candid_bytes']/baseline['total_candid_bytes']),max_query_instructions=r['max_query_instructions'],max_heap_bytes=r['max_observed_heap_bytes'],exported_hiddens=numerical,exported_states=states,decision=a,all_exported_state_bits_equal=True,report_sha256=sha(p)))
    assert outcomes
    out=dict(source_sha256=sha(Path(__file__)),scope='Successful fresh runs only; failed/tuning runs remain separately recorded in proof directory. State comparison covers exported convolution/KV and hidden, including final normalized hidden; transient client carries have different partitioning.',cases=outcomes,baseline_restored=True)
    (d/'verified.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in ['exported_hiddens','exported_states','decision']} for c in outcomes],indent=2))
if __name__=='__main__':main()
