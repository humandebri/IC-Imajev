#!/usr/bin/env python3
"""Run client-held rolled ordinary queries and compare all exports with a frozen INT8 baseline."""
import argparse,hashlib,io,json,pathlib,subprocess,sys
if not __debug__:raise RuntimeError("Proof checker requires assertions")
from proof_inputs import read_bytes,read_report,selections
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def compare(new,old,refs):
    report=json.loads((new/'report.json').read_text());before=read_report(old/'report.json',ROOT,refs)
    for key in ['model','pack_hash','input_hash']:assert report[key]==before[key],key
    assert report['replayed_queries']==0 and report['max_query_instructions']<5_000_000_000
    assert report['comparison']['typed_output_valid']
    failures=new/'queries/failures.jsonl';assert not failures.exists() or not failures.read_bytes()
    for layer in range(32):
        name=f'layer-{layer:02d}'
        if layer!=30:
            a=np.load(new/'queries'/f'{name}.npy',allow_pickle=False);b=np.load(io.BytesIO(read_bytes(old/'queries'/f'{name}.npy',ROOT,refs)),allow_pickle=False)
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),('hidden',layer)
        else:assert not (new/'queries'/f'{name}.npy').exists(),'compact tail stale hidden'
        with np.load(new/'queries/states'/f'{name}.npz',allow_pickle=False)as a,np.load(io.BytesIO(read_bytes(old/'queries/states'/f'{name}.npz',ROOT,refs)),allow_pickle=False)as b:
            assert set(a.files)==set(b.files),('state fields',layer)
            for key in a.files:assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),('state',layer,key)
    a=report['decision_query']['ok']['decision'];b=before['decision_query']['ok']['decision']
    for key in ['value','abstained','raw_logits','probabilities','unknown_probability']:assert a[key]==b[key],('decision',key)
    assert np.load(new/'final-hidden.npy',allow_pickle=False).tobytes()==np.load(io.BytesIO(read_bytes(old/'final-hidden.npy',ROOT,refs)),allow_pickle=False).tobytes()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(full_bitwise_equal=True,hidden_scope='all exported hidden; compact tail omits layer30',query_count=report['query_count'],total_instructions=report['total_instructions'],total_candid_bytes=report['total_candid_bytes'],max_query_instructions=report['max_query_instructions'],max_observed_heap_bytes=report['max_observed_heap_bytes'],wall_seconds=report['wall_seconds_this_run'],baseline_report_sha256=refs[str((old/'report.json').relative_to(ROOT))],report_sha256=sha(new/'report.json'),goal_50_verified=report['query_count']<=50)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['canister','wasm','base-proof','directory']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--join-start',action='store_true');ap.add_argument('--cases',default='617,insufficient,maximum');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);base=ROOT/a.base_proof
    paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','MODEL_LOCK.json','checkpoints/full-int8.manifest.json','scripts/run_prefix_canister.py','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};module=sha(ROOT/a.wasm);rows=[]
    flags=['--compact-lossless','--compact-heads','--fuse-add-norm','--fuse-mlp','--fuse-mlp-norm','--fuse-norm-rope','--fuse-delta','--wide-mlp','--fuse-delta-input','--reuse-projection-inputs','--fuse-attention','--fuse-delta-projected','--fuse-delta-finish','--fuse-mlp-pipeline','--fuse-attention-full','--fuse-mlp-full','--fuse-delta-full-log','--terminal-readout','--fuse-terminal-attention','--fuse-terminal-decision','--fuse-terminal-tail','--fuse-prefix-start','--roll-start']
    records={'617':0,'insufficient':19,'maximum':11}
    if a.join_start:flags.append('--join-start')
    labels=selections(a.cases,tuple(records),'cases');refs={}
    for label in labels:
        old=ROOT/f'artifacts/output-pairs-v2-{label}'
        files=[old/'report.json',old/'final-hidden.npy']+[old/'queries'/f'layer-{i:02d}.npy'for i in range(32)if i!=30]+[old/'queries/states'/f'layer-{i:02d}.npz'for i in range(32)]
        for p in files:read_bytes(p,ROOT,refs)
    for label in labels:
        params=[]
        cmd=[sys.executable,str(ROOT/'scripts/run_prefix_canister.py'),'--canister',a.canister,'--wasm',a.wasm,'--reference','artifacts/reference-serving.json','--record',str(records[label]),'--arithmetic','int8','--wire-codec','bf16-block256-exact-v1','--frame-checksum','host','--row-cap','16384','--token-cap','132','--work-cap','2500000000','--delta-head-cap','16','--attention-head-cap','8','--mlp-full-token-cap','89','--cache',str(base/'prefix/queries'),'--hybrid-cache',str(base/'packets'),'--directory',str(d/label),*flags,*params]
        (d/f'{label}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (d/f'{label}.log').open('x')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=dict(label=label,**compare(d/label,ROOT/f'artifacts/output-pairs-v2-{label}',refs));row['roll_effective']=json.loads((d/label/'report.json').read_text())['roll_effective'];rows.append(row);print(json.dumps(row),flush=True)
        (d/'partial-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
    assert sha(ROOT/a.wasm)==module
    assert all(sha(ROOT/p)==v for p,v in refs.items())
    (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,reference_hashes=refs,cases=rows),indent=2)+'\n')
if __name__=='__main__':main()
