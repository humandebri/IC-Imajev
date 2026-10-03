#!/usr/bin/env python3
"""Same-module byte decoder measurements; ordinary queries, real saved carry."""
import argparse,hashlib,json,pathlib,struct,subprocess,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',default='artifacts/prefix_codec/carry-huffman-fusion-check');ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
helper=ROOT/'artifacts/carry-planes-bench-target/debug/carry_args';wasm=ROOT/'artifacts/carry-planes-bench-target/wasm32-unknown-unknown/release/imajev_carry_planes_bench.wasm';sha=lambda b:hashlib.sha256(b).hexdigest()
sources=list((ROOT/'scripts/carry_planes_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/carry_planes_bench/Cargo.toml','scripts/carry_planes_bench/Cargo.lock','crates/imajev-runtime/src/carry_planes.rs']]+[pathlib.Path(__file__)];hashes={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in sources}
def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
assert status()==sha(wasm.read_bytes());cases=[]
for layer in [0,1,3]:
 for short in [False,True]:
  path=ROOT/a.source/f'{layer:02d}-{"short-fused-profile"if short else"fused"}.request.bin';raw=path.read_bytes();length=struct.unpack('<I',raw[:4])[0];header=json.loads(raw[4:4+length]);payload=raw[5+length:-32];n,p=header['dims'];label=f'{layer:02d}-{n}';input=d/f'{label}.payload.bin';input.write_bytes(payload);measured={}
  for optimized in [False,True]:
   arg=d/f'{label}-{optimized}.args.bin';reply=d/f'{label}-{optimized}.hex'
   subprocess.run([str(helper),'query',str(input),str(n),str(p),str(optimized).lower(),str(arg)],check=True)
   start=time.monotonic();value=subprocess.check_output(['icp','canister','call',a.canister,'measure','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);reply.write_text(value)
   m=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True,cwd=ROOT));m['wall_seconds']=time.monotonic()-start;measured['optimized'if optimized else'baseline']=m
  assert measured['optimized']['digest']==measured['baseline']['digest'];assert measured['optimized']['bytes']==measured['baseline']['bytes'];b=measured['baseline']['instructions'];c=measured['optimized']['instructions'];row=dict(layer=layer,tokens=n,prefix=p,source_frame_sha256=sha(raw),payload_sha256=sha(payload),measurements=measured,saved=b-c,change_percent=100*(c/b-1));cases.append(row);print(json.dumps(row),flush=True)
assert status()==sha(wasm.read_bytes());assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes())for p in sources}
result=dict(scope=__doc__,canister=a.canister,wasm_sha256=sha(wasm.read_bytes()),helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,ordinary_queries=12,cases=cases)
(d/'report.json').write_text(json.dumps(result,indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in sources:z.write(p,str(p.relative_to(ROOT)))
