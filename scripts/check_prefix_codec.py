#!/usr/bin/env python3
"""Actual query counter for exact hybrid prefix decoding; no full inference."""
import argparse,hashlib,json,pathlib,struct,subprocess,sys,time
import numpy as np
from explore_prefix_dense import encode,decode
ROOT=pathlib.Path(__file__).resolve().parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);ap.add_argument('--prepare-on-canister',action='store_true');ap.add_argument('--codec',choices=['huffman','nibble'],default='huffman');ap.add_argument('--directory',default='artifacts/prefix_codec/check');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 helper=ROOT/'artifacts/bounded_i16/native/release/prefix_args';wasm=ROOT/'artifacts/prefix_codec/diagnostic.wasm';expected=sha(wasm.read_bytes());cases=[]
 assert not a.prepare_on_canister or a.codec=='nibble'
 build_dir='preparation-build' if a.prepare_on_canister else ('cached-build' if a.codec=='huffman' else 'nibble-build');build=json.loads((ROOT/'artifacts/prefix_codec'/build_dir/'report.json').read_text());assert build['wasm_sha256']==expected;paths=list((ROOT/'scripts/prefix_codec_bench/src').rglob('*.rs'))+[ROOT/'scripts/prefix_codec_bench/Cargo.toml',ROOT/'scripts/prefix_codec_bench/Cargo.lock',pathlib.Path(__file__),ROOT/'scripts/explore_prefix_dense.py',ROOT/'scripts/build_prefix_codec_cached.py']
 sources={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}
 def status():return json.loads(subprocess.check_output(['icp','canister','status',a.canister,'--network','local','--identity','imajev-local','--json'],text=True))['module_hash'].removeprefix('0x')
 assert status()==expected
 capacity=json.loads((ROOT/'artifacts/prefix-dense/hybrid-capacity.json').read_text())
 for c in capacity['cases']:
  layer=c['layer'];oldpath=ROOT/f'artifacts/column32-v1-prefix/queries/states/layer-{layer:02d}.npz';logpath=ROOT/f'artifacts/output-pairs-v2-prefix/queries/states/layer-{layer:02d}.npz'
  assert sha(oldpath.read_bytes())==c['source_sha256'] and sha(logpath.read_bytes())==c['prefix_log_sha256']
  with np.load(oldpath,allow_pickle=False) as f:state=f['delta'].copy()
  with np.load(logpath,allow_pickle=False) as f:log=f['delta_log'].copy()
  if a.codec=='huffman':
   heads=c['hybrid_capacity']['original_bf16_input']['dense_heads'];assert len(heads)==20
   encoded={}
   for h in heads:
    bits=state[h].view('<u4');table,payload,mant,_=encode(bits);assert decode(table,payload,mant,bits.size,False).tobytes()==bits.tobytes()
    encoded[h]=table+struct.pack('<II',len(payload),len(mant))+payload+mant
   magic=b'HPF1'
  else:
   encoded={}
   for h in range(32):
    bits=state[h].view('<u4').reshape(-1);exps=((bits>>23)&255).astype(np.uint8);hist=np.bincount(exps,minlength=256)
    base=max(range(241),key=lambda b:int(hist[b:b+15].sum()));inside=(exps>=base)&(exps<base+15)
    codes=np.where(inside,exps.astype(np.int16)-base,15).astype(np.uint8)
    nib=codes[::2]|(codes[1::2]<<4);exceptions=exps[~inside].tobytes();low=(bits&0x7fffff)|((bits>>31)<<23)
    mant=np.column_stack([(low>>shift).astype(np.uint8) for shift in [0,8,16]]).tobytes()
    encoded[h]=bytes([base,0,0,0])+struct.pack('<I',len(exceptions))+nib.tobytes()+mant+exceptions
   log_pair=45*(128*2+2*128*4+2*4);total=16424+2*(87*2560+3*8192)+12+16*log_pair;heads=[]
   for pair in sorted(range(16),key=lambda p:len(encoded[2*p])+len(encoded[2*p+1])):
    delta=len(encoded[2*pair])+len(encoded[2*pair+1])-log_pair
    if total+delta<=2000000:total+=delta;heads.extend([2*pair,2*pair+1])
   heads.sort();assert len(heads)>=18;magic=b'NPF1'
  mask=sum(1<<h for h in heads);packet=bytearray(magic+struct.pack('<II',45,mask))
  for h in heads:packet.extend(encoded[h])
  remaining=[h for h in range(32) if h not in heads];pairs=sorted(set(h//2 for h in remaining))
  keys=log[:45*2048].reshape(45,16,128)[:,pairs,:];assert np.all((keys.view(np.uint32)&65535)==0)
  packet.extend((keys.view(np.uint32)>>16).astype('<u2').tobytes())
  updates=log[45*2048:45*(2048+4096)].reshape(45,32,128)[:,remaining,:];decays=log[45*(2048+4096):].reshape(45,32)[:,remaining]
  packet.extend(updates.astype('<f4').tobytes());packet.extend(decays.astype('<f4').tobytes())
  hybrid_path=d/f'layer-{layer:02d}.hybrid.bin';old_file=d/f'layer-{layer:02d}.state.bin';log_file=d/f'layer-{layer:02d}.log.bin'
  assert len(packet)+16424+2*(87*2560+3*8192)<=2000000
  hybrid_path.write_bytes(packet);old_file.write_bytes(state.astype('<f4').tobytes());log_file.write_bytes(log.astype('<f4').tobytes())
  preparation=None
  if a.prepare_on_canister:
   arg=d/f'layer-{layer:02d}-prepare.args.bin';subprocess.run([str(helper),'prepare',str(log_file),str(arg)],check=True)
   begin=time.monotonic();reply=subprocess.check_output(['icp','canister','call',a.canister,'prepare_prefix','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True);elapsed=time.monotonic()-begin
   rp=d/f'layer-{layer:02d}-prepare.reply.hex';rp.write_text(reply)
   preparation=json.loads(subprocess.check_output([str(helper),'decode-preparation',str(rp),str(hybrid_path)],text=True))
   assert hybrid_path.read_bytes()==packet, 'Canister preparation differs from independent reference encoding'
   assert preparation['digest']==list(bytes.fromhex(sha(old_file.read_bytes())))
   preparation.update(wall_seconds=elapsed,request_candid_bytes=arg.stat().st_size,reply_candid_bytes=len(reply.strip().removeprefix('0x'))//2,reply_sha256=sha(reply.encode()))
  native=json.loads(subprocess.check_output([str(helper),'native',str(hybrid_path),str(old_file)],text=True));oracle=list(bytes.fromhex(sha(old_file.read_bytes())));assert native['digest']==oracle
  measures={}
  for hybrid,path in [(False,log_file),(True,hybrid_path)]:
   arg=d/f'layer-{layer:02d}-{hybrid}.args.bin';subprocess.run([str(helper),'query',str(hybrid).lower(),str(path),str(arg)],check=True)
   begin=time.monotonic();reply=subprocess.check_output(['icp','canister','call',a.canister,'restore','--network','local','--identity','imajev-local','--query','--args-file',str(arg),'--args-format','bin','--output','hex'],text=True);elapsed=time.monotonic()-begin
   rp=d/f'layer-{layer:02d}-{hybrid}.reply.hex';rp.write_text(reply);measured=json.loads(subprocess.check_output([str(helper),'decode',str(rp)],text=True));assert measured['digest']==oracle
   measured.update(wall_seconds=elapsed,request_candid_bytes=arg.stat().st_size,reply_sha256=sha(reply.encode()));measures['hybrid' if hybrid else 'baseline']=measured
  row=dict(layer=layer,dense_heads=heads,source_state_sha256=c['source_sha256'],source_log_sha256=c['prefix_log_sha256'],hybrid_payload_sha256=sha(packet),hybrid_payload_bytes=len(packet),measurements=measures,preparation=preparation,native=native,change_percent=100*(measures['hybrid']['instructions']/measures['baseline']['instructions']-1))
  cases.append(row);print(json.dumps(dict(layer=layer,change_percent=row['change_percent'],baseline=measures['baseline']['instructions'],hybrid=measures['hybrid']['instructions'])),flush=True)
 assert status()==expected and all(sha(p.read_bytes())==sources[str(p.relative_to(ROOT))] for p in paths)
 r=dict(codec=a.codec,canister_prefix_preparation=a.prepare_on_canister,scope='Exact prefix reconstruction only; no projection/recurrence/suffix inference, judgment or total-query-count improvement claim. Counters exclude digest/CDK Candid and include all decoder work. No question state is retained in canister.',canister=a.canister,wasm_sha256=expected,source_hashes=sources,cached_build=build,helper_sha256=sha(helper.read_bytes()),capacity_sha256=sha((ROOT/'artifacts/prefix-dense/hybrid-capacity.json').read_bytes()),ordinary_queries=(72 if a.prepare_on_canister else 48),cases=cases)
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
