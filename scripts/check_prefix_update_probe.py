#!/usr/bin/env python3
"""Compare both restore Wasms with every recorded exact 32-head prefix state."""
import argparse,hashlib,json,pathlib,subprocess,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
for name in ['baseline-canister','hoisted-canister','build','directory']:ap.add_argument('--'+name,required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);assert not(d/'report.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=ROOT/a.build;b=json.loads((build/'report.json').read_text());source=dict(b['source_hashes']);source[str(pathlib.Path(__file__).relative_to(ROOT))]=sha(pathlib.Path(__file__))
helper=ROOT/'artifacts/bounded_i16/native/release/prefix_args';source[str(helper.relative_to(ROOT))]=sha(helper)
capacity_path=ROOT/'artifacts/prefix-dense/hybrid-capacity.json';source[str(capacity_path.relative_to(ROOT))]=sha(capacity_path)
assert all(sha(ROOT/p)==v for p,v in source.items())
expected={r['variant']:r['wasm_sha256']for r in b['builds']};targets={'baseline':a.baseline_canister,'hoisted':a.hoisted_canister}
def status(principal):return json.loads(subprocess.check_output(['icp','canister','status',principal,'--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
assert all(status(v)==expected[k]for k,v in targets.items())
references={};rows=[]
for c in json.loads(capacity_path.read_text())['cases']:
 layer=c['layer'];old=ROOT/f'artifacts/column32-v1-prefix/queries/states/layer-{layer:02d}.npz';logfile=ROOT/f'artifacts/output-pairs-v2-prefix/queries/states/layer-{layer:02d}.npz'
 assert sha(old)==c['source_sha256']and sha(logfile)==c['prefix_log_sha256'];references[str(old.relative_to(ROOT))]=sha(old);references[str(logfile.relative_to(ROOT))]=sha(logfile)
 with np.load(old,allow_pickle=False)as z:state=z['delta'].astype('<f4').reshape(32,128,128)
 with np.load(logfile,allow_pickle=False)as z:raw=z['delta_log'].astype('<f4').tobytes()
 assert len(raw)==45*6176*4
 f=d/f'layer-{layer:02d}.log.bin';f.write_bytes(raw)
 for layout,key_major in [('value_major',False),('key_major',True)]:
  oracle=hashlib.sha256((state.transpose(0,2,1)if key_major else state).tobytes()).digest()
  arg=d/f'layer-{layer:02d}-{layout}.args.bin';subprocess.run([str(helper),'query',str(key_major).lower(),str(f),str(arg)],check=True)
  measures={}
  for variant,canister in targets.items():
   begin=time.monotonic();reply=subprocess.check_output(['icp','canister','call',canister,'restore','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True)
   out=d/f'layer-{layer:02d}-{layout}-{variant}.reply.hex';out.write_text(reply);m=json.loads(subprocess.check_output([str(helper),'decode',str(out)],text=True));assert bytes(m['digest'])==oracle and m['values']==32*128*128
   m.update(wall_seconds=time.monotonic()-begin,request_candid_bytes=arg.stat().st_size,reply_candid_bytes=len(reply.strip().removeprefix('0x'))//2);measures[variant]=m
  row=dict(layer=layer,layout=layout,bitwise_digest_equal=True,measurements=measures,instructions_saved=measures['baseline']['instructions']-measures['hoisted']['instructions']);rows.append(row);print(json.dumps(dict(layer=layer,layout=layout,instructions_saved=row['instructions_saved'])),flush=True)
assert all(status(v)==expected[k]for k,v in targets.items())and all(sha(ROOT/p)==v for p,v in source.items())and all(sha(ROOT/p)==v for p,v in references.items())
(d/'report.json').write_text(json.dumps(dict(module_hashes=expected,targets=targets,source_hashes=source,reference_hashes=references,cases=rows,ordinary_queries=2*len(rows),scope='Prefix restoration only; no full inference or query-count claim',total_restore_instructions_saved=sum(r['instructions_saved']for r in rows)),indent=2)+'\n')
with zipfile.ZipFile(d/'validated-source.zip','x',zipfile.ZIP_DEFLATED)as z:
 for p in source:
  if not p.startswith('artifacts/'):z.write(ROOT/p,p)
