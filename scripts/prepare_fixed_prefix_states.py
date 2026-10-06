#!/usr/bin/env python3
"""Prepare 24 exact immutable prefix states, excluding question inference."""
import argparse,hashlib,json,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,decode
from prefix_inference import verify_module
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--wasm',required=True);ap.add_argument('--directory',required=True);a=ap.parse_args();d=ROOT/a.directory;d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/query-packing-v3/prefix-v2';packets=ROOT/'artifacts/query-packing-v3/packets-v2'
 cache=json.loads((packets/'cache.json').read_text());r=json.loads((source/'report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
 assert r['model']==m['model']==sha(ROOT/'MODEL_LOCK.json');assert r['tokens']==27 and r['pack_hash']==m['pack_hash']
 assert cache['identity']['source_report_sha256']==sha(source/'report.json')
 helper=ROOT/'artifacts/prefix-state-cache-v1/prefix-state-args';module=sha(ROOT/a.wasm)
 paths=[Path(__file__),ROOT/'scripts/prefix_state_args.rs',helper,source/'report.json',packets/'cache.json',ROOT/a.wasm,ROOT/'client/transport.py',ROOT/'client/prefix_inference.py',ROOT/'artifacts/query-packing-v3/build/imajev-client']
 metrics=[q for q in r['queries'] if q.get('op')=='delta_full_log_integer'];assert len(metrics)==24
 for q in metrics:
  layer=int(q['tensor'].split('.')[3]);paths+=[source/'queries'/f'{q["index"]:06d}.response.bin',packets/f'layer-{layer:02d}.npf1']
 identities={str(p.relative_to(ROOT)):sha(p) for p in paths}
 def verify():
  t=Transport(m['model'],'http://localhost:8001/',a.canister,str(ROOT/'artifacts/imajev-local.pem'),d/'module',m['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
  try:verify_module(t,module)
  finally:t.close()
 verify();did=d/'prepare.did';did.write_text('service:{prepare_fixed_prefix_state:(vec nat8,vec nat8)->(variant {Ok:record {nat32;nat64;nat64};Err:text})}')
 rows=[]
 for number,q in enumerate(metrics,1):
  layer=int(q['tensor'].split('.')[3]);frame=source/'queries'/f'{q["index"]:06d}.response.bin';header,values=decode(frame.read_bytes());assert header['dims']==[27,32,0,1]
  log=values[27*2560+24576:].astype('<f4').tobytes();assert len(log)==27*6176*4
  lp=d/f'layer-{layer:02d}.log.f32';lp.write_bytes(log);packet=packets/f'layer-{layer:02d}.npf1';arg=d/f'{layer:02d}.args.bin';reply=d/f'{layer:02d}.reply.hex'
  subprocess.run([str(helper),'args',str(lp),str(packet),str(arg)],check=True);assert arg.stat().st_size<1_990_000
  start=time.monotonic();raw=subprocess.check_output(['icp','canister','call',a.canister,'prepare_fixed_prefix_state','--network','local','--identity','imajev-local','--candid',str(did),'--args-file',str(arg),'--args-format','bin','--output','hex'],text=True);reply.write_text(raw)
  result=json.loads(subprocess.check_output([str(helper),'decode',str(reply)],text=True));assert result['count']==number and result['bytes']==number*2_097_152
  row=dict(layer=layer,source_frame_sha256=sha(frame),log_sha256=sha(lp),packet_sha256=sha(packet),args_sha256=sha(arg),reply_sha256=sha(reply),result=result,wall_seconds=time.monotonic()-start);rows.append(row);print(json.dumps(row),flush=True)
 verify();assert identities=={p:sha(ROOT/p) for p in identities}
 report=dict(wasm_sha256=module,source_hashes=identities,update_calls=len(rows),state_bytes=rows[-1]['result']['bytes'],instructions=sum(x['result']['instructions'] for x in rows),layers=rows,scope='Owner prepares fixed prefix state only. Raw log and hybrid packet are restored and bit-compared before caching. Updates and preparation cost excluded from per-question inference.')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
if __name__=='__main__':main()
