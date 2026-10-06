#!/usr/bin/env python3
"""Measure register-resident vs frozen key-major Delta and a native F32 oracle."""
import argparse,hashlib,json,subprocess,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/delta-register-v3/check';B=ROOT/'artifacts/delta-register-v3/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native(n):
 def seq(length,seed,scale):
  s=seed;values=[];factor=np.float32(np.float32(scale)/np.float32(32768))
  for _ in range(length):
   s=(s*1664525+1013904223)&0xffffffff;values.append(np.float32(np.float32((s>>16)-32768)*factor))
  return np.array(values,dtype='<f4')
 q=seq(n*128,17,.08).reshape(n,128);k=seq(n*128,53,.08).reshape(n,128);v=seq(n*128,97,.2).reshape(n,128)
 g=seq(n,31,.15)+np.float32(.85);b=seq(n,73,.45)+np.float32(.5)
 state=seq(16384,41,.1).reshape(128,128).T.copy();out=[]
 for t in range(n):
  # Explicit F32 operation boundaries and ascending key order; no matrix BLAS.
  decayed=state*g[t];mem=np.zeros(128,dtype=np.float32)
  for i in range(128):mem=mem+decayed[i]*k[t,i]
  update=(v[t]-mem)*b[t];o=np.zeros(128,dtype=np.float32)
  for i in range(128):
   state[i]=decayed[i]+k[t,i]*update;o=o+state[i]*q[t,i]
  out.append(o)
 return hashlib.sha256(np.array(out,dtype='<f4').tobytes()+state.astype('<f4').tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);a=ap.parse_args();D.mkdir(exist_ok=False)
 build=json.loads((B/'report.json').read_text());identities=build['source_hashes']|build['dependency_hashes']
 helper=ROOT/'artifacts/delta-writeback-target/debug/writeback_args'
 identities|={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(helper.relative_to(ROOT)):sha(helper)}
 assert all(sha(ROOT/p)==h for p,h in identities.items())
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
 assert status()==build['wasm_sha256']
 did=D/'diagnostic.did';did.write_text('''type M=record {digest:vec nat8;kernel_instructions:nat64;handler_instructions:nat64;values:nat64;scratch_unchanged:bool};service:{measure_layout:(nat32,nat32,nat32,bool)->(M) query}''')
 rows=[]
 for n in [1,8,57,59,67,87,132]:
  measurements={};expected=native(n)
  for candidate in [False,True]:
   label=f'{n}-{candidate}';arg=D/f'{label}.args.bin';reply=D/f'{label}.hex'
   subprocess.run([str(helper),'query',str(n),'128','128',str(candidate).lower(),str(arg)],check=True)
   start=time.monotonic();raw=subprocess.check_output(['icp','canister','call',a.canister,'measure_layout','--network','local','--identity','imajev-local','--query','--candid',str(did),'--args-file',str(arg),'--args-format','bin','--output','hex'],text=True)
   reply.write_text(raw);m=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True));m.update(reply=str(reply.relative_to(ROOT)),reply_sha256=sha(reply),wall_seconds=time.monotonic()-start)
   assert bytes(m['digest']).hex()==expected,(n,candidate,m['digest'],expected)
   measurements['register' if candidate else 'baseline']=m
  before=measurements['baseline']['kernel_instructions'];after=measurements['register']['kernel_instructions'];row=dict(tokens=n,native_digest=expected,bitwise_equal=True,measurements=measurements,reduction_percent=100*(1-after/before));rows.append(row);print(json.dumps(dict(tokens=n,before=before,after=after,reduction_percent=row['reduction_percent'])),flush=True)
 assert status()==build['wasm_sha256'];assert all(sha(ROOT/p)==h for p,h in identities.items())
 report=dict(wasm_sha256=build['wasm_sha256'],source_hashes=identities,candid_sha256=sha(did),canister=a.canister,ordinary_queries=2*len(rows),scope='Synthetic single 128x128 Delta head. Equal key-major initial state. Both preserve the final key-major state. Output and final-state bits are included in the digest. Kernel counters include output allocation, register load and computation; handler includes common input generation and transpose. Native oracle uses explicit numpy F32 operations and key order. Full model accuracy/calls not tested.',cases=rows)
 (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in identities:z.write(ROOT/p,p)
if __name__=='__main__':main()
