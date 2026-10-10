#!/usr/bin/env python3
"""Profile frozen ordinary-query requests without altering inference journals."""
import argparse,hashlib,json,pathlib,sys,zipfile
if not __debug__:raise RuntimeError('Proof checker requires assertions')
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_bytes,selections
from transport import Transport,decode
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['canister','wasm','roll','standard','directory']:ap.add_argument('--'+name,required=True)
 ap.add_argument('--roll-indices',default='0,1,2,3,4,5,6')
 ap.add_argument('--standard-indices',default='0,1,2,3,4,5,6,7')
 a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 if any(d.iterdir()):raise ValueError('Use a new evidence directory')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=list((ROOT/'client').glob('*.py'))+list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'canisters/inference/src').rglob('*.rs'))+[ROOT/p for p in ['MODEL_LOCK.json','checkpoints/full-int8.manifest.json','Cargo.toml','Cargo.lock','crates/imajev-runtime/Cargo.toml','canisters/inference/Cargo.toml','scripts/proof_inputs.py','scripts/build_full_prefix_candidate.py']]+[pathlib.Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());module=sha(ROOT/a.wasm)
 refs={};inputs=[]
 for label,folder,selected in [('roll',a.roll,a.roll_indices),('standard',a.standard,a.standard_indices)]:
  source=ROOT/folder/'queries'
  allowed=tuple(str(int(p.name.split('.')[0]))for p in source.glob('*.request.bin')if p.name.split('.')[0].isdigit())
  for item in selections(selected,allowed,label+' indices'):
   index=int(item);request=source/f'{index:06d}.request.bin';response=source/f'{index:06d}.response.bin'
   raw=read_bytes(request,ROOT,refs);expected=decode(read_bytes(response,ROOT,refs))[1].tobytes()
   # Request payloads can differ from reply payloads for the same operation.
   # The canister validates the complete frozen request; only read its header here.
   headlen=int.from_bytes(raw[:4],'little')
   if len(raw)<36 or not 0<headlen<=16384 or 4+headlen>len(raw)-32:raise ValueError('profile request header bounds')
   h=json.loads(raw[4:4+headlen])
   # Pass the exact frozen bytes to the bridge, never reread a mutable journal.
   frozen=d/f'{label}-{index:06d}.request.bin';frozen.write_bytes(raw)
   inputs.append((label,index,request,response,frozen,h,expected))
 t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash']);rows=[]
 try:
  verify_module(t,module)
  for label,index,request,response,frozen,h,expected in inputs:
   output=d/f'{label}-{index:06d}.response.bin';result=t.command(dict(diagnostics=True,op='profile',input=str(frozen),output=str(output)))
   assert decode(output.read_bytes())[1].tobytes()==expected
   row=dict(label=label,index=index,op=h['op'],tensor=h.get('tensor'),request_sha256=refs[str(request.relative_to(ROOT))],response_sha256=refs[str(response.relative_to(ROOT))],profile=result,bitwise_equal=True);rows.append(row);print(json.dumps(row),flush=True)
  verify_module(t,module)
  assert sha(ROOT/a.wasm)==module and hashes=={str(p.relative_to(ROOT)):sha(p)for p in paths}
  assert all(sha(ROOT/p)==v for p,v in refs.items())
  (d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,reference_hashes=refs,profile_query_count=len(rows),production_inference_query_count=0,cases=rows),indent=2)+'\n')
  with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
   for p in paths:z.write(p,str(p.relative_to(ROOT)))
 finally:t.close()
if __name__=='__main__':main()
