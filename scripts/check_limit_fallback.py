#!/usr/bin/env python3
"""Force an experimental chunk limit on real input; verify isolated standard restart."""
import argparse,hashlib,json,pathlib,subprocess,sys
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from check_roll_graph import compare
from proof_inputs import read_bytes

def main():
 ap=argparse.ArgumentParser()
 for key in ['base-proof','directory','canister','wasm']:ap.add_argument('--'+key,required=True)
 a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 if any(d.iterdir()):raise ValueError('fresh directory required')
 paths=list((ROOT/'client').glob('*.py'))+[ROOT/'scripts/run_prefix_canister.py',pathlib.Path(__file__)];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};module=sha(ROOT/a.wasm)
 original=json.loads((ROOT/'artifacts/tail50/tail-proof-v1/617.command.json').read_text())
 for flag,value in [('--canister',a.canister),('--wasm',a.wasm),('--cache',str((ROOT/a.base_proof/'prefix/queries').resolve())),('--hybrid-cache',str((ROOT/a.base_proof/'packets').resolve())),('--directory',str((d/'run').resolve())),('--tail-front28','8192')]:original[original.index(flag)+1]=value
 (d/'command.json').write_text(json.dumps(original,indent=2)+'\n')
 refs={};old=ROOT/'artifacts/output-pairs-v2-617'
 for p in [old/'report.json',old/'final-hidden.npy']+[old/'queries'/f'layer-{i:02d}.npy'for i in range(32)if i!=30]+[old/'queries/states'/f'layer-{i:02d}.npz'for i in range(32)]:read_bytes(p,ROOT,refs)
 with (d/'run.log').open('w')as f:subprocess.run(original,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 r=json.loads((d/'run/report.json').read_text());event=r['fallback'];child=d/'run/standard-fallback'
 assert event['stage']=='query' and event['failed_query_count']==1
 assert r['total_instructions']is None and r['total_candid_bytes']is None and r['unmeasured_failed_queries']==1
 assert r['query_count']==len(event['primary_queries'])+1+62
 assert (d/'run/queries'/f"{len(event['primary_queries']):06d}.request.bin").exists()
 proof=compare(child,old,refs);assert proof['query_count']==62
 # Resume selects the saved fallback marker, not the failed primary query.
 with (d/'resume.log').open('w')as f:subprocess.run(original,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 resumed=json.loads((d/'run/report.json').read_text());assert resumed['executed_query_count']==0 and resumed['query_count']==r['query_count']
 assert json.loads((child/'report.json').read_text())['replayed_queries']==62
 assert hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths} and sha(ROOT/a.wasm)==module and all(sha(ROOT/p)==v for p,v in refs.items())
 (d/'report.json').write_text(json.dumps(dict(source_hashes=hashes,reference_hashes=refs,module_sha256=module,standard_bitwise_equal=proof,attempt_query_count=r['query_count'],fallback_resume_executed=0),indent=2)+'\n');print('fallback and resume verified',r['query_count'])
if __name__=='__main__':main()
