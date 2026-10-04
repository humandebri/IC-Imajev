#!/usr/bin/env python3
"""Measure actual residual byte-plane decoding; no model arithmetic on the host."""
import argparse,hashlib,json,pathlib,struct,subprocess,sys,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from proof_inputs import read_report,residual_cases
from transport import decode
from mlp_delta_carry import encode_plane,encode_dictionary_plane
ap=argparse.ArgumentParser(description=__doc__)
for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
ap.add_argument("--interleave",action="store_true");a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not(d/'report.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[ROOT/'scripts/proof_inputs.py',pathlib.Path(__file__),ROOT/'client/mlp_delta_carry.py',ROOT/'client/transport.py',ROOT/'crates/imajev-runtime/src/carry_planes.rs',ROOT/'scripts/prefix_update_bench/src/lib.rs']
helper=ROOT/'artifacts/bounded_i16/native/release/prefix_args';paths.append(helper);hashes={str(p.relative_to(ROOT)):sha(p)for p in paths};module=sha(ROOT/a.wasm)
def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
assert status()==module
base=ROOT/'artifacts/prefix_codec/attention-q4-all-v2';ref=base/'report.json';references={};cases=residual_cases(read_report(ref,ROOT,references));rows=[]
for case in cases:
 p=base/f"{case['call']['index']:06d}.response.bin";references[str(p.relative_to(ROOT))]=sha(p);h,x=decode(p.read_bytes());n=case['tokens'];raw=(x[:n*2560].astype('<f4').view('<u4')>>16).astype('<u2').tobytes();oracle=hashlib.sha256(raw[::2]+raw[1::2]).digest()
 if a.interleave:oracle=hashlib.sha256(raw).digest()
 measures={}
 for mode in (['legacy','simd']if a.interleave else['huffman','dictionary']):
  if a.interleave:packet=struct.pack('<I',n*2560)+raw[::2]+raw[1::2]
  else:packet=struct.pack('<I',n*2560)+encode_plane(raw[::2],.1)+(encode_plane(raw[1::2],.1)if mode=='huffman'else encode_dictionary_plane(raw[1::2]))
  scope=f"{case['label']}-l{case['layer']}-b{case['begin']}-{mode}";input=d/f'{scope}.bin';input.write_bytes(packet);arg=d/f'{scope}.args.bin';subprocess.run([str(helper),'query',str(mode!='simd').lower(),str(input),str(arg)],check=True)
  start=time.monotonic();reply=subprocess.check_output(['icp','canister','call',a.canister,'interleave_residual'if a.interleave else'decode_residual','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True);out=d/f'{scope}.reply.hex';out.write_text(reply)
  m=json.loads(subprocess.check_output([str(helper),'decode',str(out)],text=True));assert bytes(m['digest'])==oracle and m['values']==n*2560*2
  m.update(wall_seconds=time.monotonic()-start,request_candid_bytes=arg.stat().st_size,reply_candid_bytes=len(reply.strip().removeprefix('0x'))//2);measures[mode]=m
 row=dict(label=case['label'],layer=case['layer'],front=case['begin'],tokens=n,byte_digest_equal=True,measurements=measures,instructions_saved=measures['legacy'if a.interleave else'huffman']['instructions']-measures['simd'if a.interleave else'dictionary']['instructions']);rows.append(row);print(json.dumps({k:row[k]for k in ['label','layer','front','instructions_saved']}),flush=True)
assert status()==module and all(sha(ROOT/p)==v for p,v in hashes.items())and all(sha(ROOT/p)==v for p,v in references.items())
(d/'report.json').write_text(json.dumps(dict(module_sha256=module,source_hashes=hashes,reference_hashes=references,cases=rows,ordinary_queries=2*len(rows),scope='Residual interleave only'if a.interleave else'Residual byte decoding only; full inference and query reduction unverified'),indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
 for p in paths:
  if not str(p.relative_to(ROOT)).startswith('artifacts/'):z.write(p,str(p.relative_to(ROOT)))
