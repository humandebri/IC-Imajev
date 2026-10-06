#!/usr/bin/env python3
"""Measure optional direct key-major state consumption using ordinary queries.
Synthetic inputs test the kernel; the full-model proof is separate.
"""
import argparse,hashlib,json,pathlib,subprocess,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args()
d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
helper=ROOT/'artifacts/delta-writeback-target/debug/writeback_args'
wasm=ROOT/'artifacts/delta-writeback-target/wasm32-unknown-unknown/release/imajev_delta_writeback_bench.wasm'
sha=lambda b:hashlib.sha256(b).hexdigest()
sources=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))+[ROOT/'crates/inference-core/Cargo.toml']+list((ROOT/'scripts/delta_writeback_bench/src').rglob('*.rs'))+[ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'scripts/delta_writeback_bench/Cargo.toml',ROOT/'scripts/delta_writeback_bench/Cargo.lock',ROOT/'Cargo.toml',ROOT/'Cargo.lock',pathlib.Path(__file__)]
hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True,cwd=ROOT))['module_hash'].removeprefix('0x')
expected=sha(wasm.read_bytes());assert status()==expected
cases=[]
for n,dk,dv in [(n,128,128) for n in [1,7,8,32,45,80,87,89,132]]+[(7,3,4),(45,128,12),(87,128,16)]:
 measured={}
 for discard in [False,True]:
  label=f'{n}-{dk}-{dv}-{discard}';arg=d/f'{label}.args.bin';reply=d/f'{label}.hex'
  subprocess.run([str(helper),'query',str(n),str(dk),str(dv),str(discard).lower(),str(arg)],check=True)
  start=time.monotonic();raw=subprocess.check_output(['icp','canister','call',a.canister,'measure_layout','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True,cwd=ROOT);elapsed=time.monotonic()-start;reply.write_text(raw)
  m=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True,cwd=ROOT));m['wall_seconds']=elapsed;measured['key_major' if discard else 'baseline']=m
 assert measured['key_major']['digest']==measured['baseline']['digest'];assert not measured['key_major']['scratch_unchanged'];assert measured['baseline']['scratch_unchanged']
 b=measured['baseline']['kernel_instructions'];c=measured['key_major']['kernel_instructions']
 row=dict(tokens=n,dk=dk,dv=dv,measurements=measured,kernel_saved=b-c,kernel_change_percent=100*(c/b-1));cases.append(row);print(json.dumps(row),flush=True)
assert status()==expected;assert hashes=={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sources}
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in sources:z.write(p,str(p.relative_to(ROOT)))
(d/'report.json').write_text(json.dumps(dict(canister=a.canister,wasm_sha256=expected,helper_sha256=sha(helper.read_bytes()),source_hashes=hashes,ordinary_queries=2*len(cases),scope='Synthetic key-major vs value-major Delta-only Wasm bitwise parity. Both discard final state. Kernel counters exclude initial layout preparation, input generation, digest and Candid encoding; handler counters charge conversion for the diagnostic. Full-model packet decoder and log restoration cost is measured separately.',cases=cases),indent=2)+'\n')
