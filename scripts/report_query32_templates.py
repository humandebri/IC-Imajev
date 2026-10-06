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
    build=json.loads((ROOT/'artifacts/voting-template-prefix-v1/full-build/report.json').read_text())
    for kind in ['source_hashes','dependency_hashes']:
        for path,h in build[kind].items():assert sha(ROOT/path)==h,path
    assert len({p['function_index'] for p in build['patches']})==len(build['patches'])
    full=json.loads((d/'report.json').read_text());assert full['complete'] and full['ordinary_queries']==96 and full['baseline_snapshot_restored'] and full['snapshot_deleted']
    bank=ROOT/'artifacts/voting-template-prefix-v1';metadata=json.loads((bank/'template.json').read_text())
    assert metadata['tokens']==38 and metadata['stem'].endswith(' to ') and 'Question:' not in metadata['stem']
    import zipfile
    with zipfile.ZipFile(bank/'source.zip') as z:
        for path,h in metadata['source_hashes'].items():assert hashlib.sha256(z.read(path)).hexdigest()==h,path
    from prefix_inference import load_cache
    manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
    prefix=load_cache(bank/'prefix/queries',manifest,full['baseline'])
    assert prefix['metadata']['token_ids']==metadata['token_ids']
    from prefix_hybrid import load_packets
    prefix['hybrid_packets']=load_packets(bank/'packets',prefix)
    preparation=json.loads((bank/'prefix/report.json').read_text());packet_preparation=json.loads((bank/'packet-preparation.json').read_text())
    assert preparation['prefix_mode']=='prepare' and preparation['tokens']==38 and preparation['replayed_queries']==0 and preparation['query_count']==66
    assert packet_preparation['preparation_queries']==24 and not packet_preparation['cache_hit']
    outcomes=[]
    for proposal in ['617','620','653']:
        p=d/proposal/'report.json'
        if not p.exists():continue
        r=json.loads(p.read_text());old=ROOT/f'artifacts/boomdao-current-v1/{proposal}-r1';baseline=json.loads((old/'report.json').read_text())
        assert r['query_count']==r['executed_query_count']==len(r['queries'])==32 and r['replayed_queries']==0
        assert not (p.parent/'fallback.json').exists() and not (p.parent/'queries/failures.jsonl').exists()
        assert r['input_hash']==baseline['input_hash'] and r['model']==baseline['model'] and r['pack_hash']==baseline['pack_hash']
        assert r['wasm_sha256']==sha(ROOT/'artifacts/voting-template-prefix-v1/full-build/full.wasm')
        expected_prefix=27 if proposal=='653' else 38
        assert r['tokens']==baseline['tokens'] and r['processed_tokens']==r['tokens']-expected_prefix
        assert r['input_hash']==baseline['input_hash']
        numerical=[]
        for file in sorted((p.parent/'queries').glob('layer-*.npy')):
            ref=old/'queries'/file.name
            assert ref.exists() and np.array_equal(np.load(file).view('<u4'),np.load(ref).view('<u4')),file
            numerical.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),reference=str(ref.relative_to(ROOT))))
        assert len(numerical)==31
        if proposal!='653':
            for layer in range(30):assert prefix['hidden'][layer].tobytes()==np.load(old/f'queries/layer-{layer:02d}.npy')[:38].tobytes(),(proposal,layer)

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
        outcomes.append(dict(proposal=proposal,prefix_tokens=expected_prefix,suffix_tokens=r['processed_tokens'],query_count=32,baseline_query_count=39,total_instructions=r['total_instructions'],instruction_reduction_percent=100*(1-r['total_instructions']/baseline['total_instructions']),candid_bytes=r['total_candid_bytes'],candid_reduction_percent=100*(1-r['total_candid_bytes']/baseline['total_candid_bytes']),max_query_instructions=r['max_query_instructions'],max_heap_bytes=r['max_observed_heap_bytes'],exported_hiddens=numerical,exported_states=states,decision=a,all_exported_state_bits_equal=True,report_sha256=sha(p)))
    assert outcomes
    out=dict(source_sha256=sha(Path(__file__)),scope='Successful fresh runs only; failed/tuning runs remain separately recorded in proof directory. State comparison covers exported convolution/KV and hidden, including final normalized hidden; transient client carries have different partitioning.',cases=outcomes,baseline_restored=True,fixed_voting_prefix_preparation_queries=66,fixed_voting_packet_preparation_queries=24,fixed_voting_prefix_preparation_instructions=preparation['total_instructions'],fixed_weight_preparation_updates=full['preparation']['update_calls_this_run'],fixed_prefix_state_preparation_updates=full['fixed_prefix_preparation']['update_calls'],per_inference_updates=0)
    (d/'verified.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in ['exported_hiddens','exported_states','decision']} for c in outcomes],indent=2))
if __name__=='__main__':main()
