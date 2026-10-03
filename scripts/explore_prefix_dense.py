#!/usr/bin/env python3
"""Lossless dense prefix capacity probe. No host inference or canister mutation.
Measures actual reversible F32 encodings; request budgets are estimates only.
"""
import argparse,hashlib,heapq,json,pathlib,zlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'client'))
from transport import decode as decode_frame
def huffman_lengths(hist):
 heap=[(int(n),i,{i:0}) for i,n in enumerate(hist) if n]
 heapq.heapify(heap);serial=256
 if len(heap)==1:return {heap[0][1]:1}
 while len(heap)>1:
  n,_,a=heapq.heappop(heap);m,_,b=heapq.heappop(heap)
  c={k:v+1 for k,v in (a|b).items()};heapq.heappush(heap,(n+m,serial,c));serial+=1
 return heap[0][2]
def encode(bits):
 # Mantissa plus sign occupies exactly 24 bits; exponent is encoded separately.
 flat=bits.reshape(-1);low=(flat&0x7fffff)|((flat>>31)<<23)
 mant=np.column_stack([(low>>shift).astype(np.uint8) for shift in [0,8,16]])
 exp=((flat>>23)&255).astype(np.uint8)
 lengths=huffman_lengths(np.bincount(exp,minlength=256));codes={};code=0;previous=0
 for symbol,length in sorted(lengths.items(),key=lambda a:(a[1],a[0])):
  code<<=length-previous;codes[symbol]=(code,length);code+=1;previous=length
 out=bytearray();acc=0;used=0
 for symbol in exp:
  code,length=codes[int(symbol)];acc=(acc<<length)|code;used+=length
  while used>=8:used-=8;out.append((acc>>used)&255);acc&=(1<<used)-1
 if used:out.append(acc<<(8-used))
 table=bytes(lengths.get(i,0) for i in range(256))
 # Two actual complete encodings, both decoded independently below.
 raw=mant.tobytes();planes=mant.T.copy().tobytes();compressed=zlib.compress(planes,1)
 return table,bytes(out),raw,compressed

def decode(table,payload,mant,count,compressed):
 if compressed:
  raw=zlib.decompress(mant);assert len(raw)==count*3
  m=np.frombuffer(raw,np.uint8).reshape(3,count).T
 else:m=np.frombuffer(mant,np.uint8).reshape(count,3)
 low=m[:,0].astype(np.uint32)|(m[:,1].astype(np.uint32)<<8)|(m[:,2].astype(np.uint32)<<16)
 trie={};code=0;previous=0
 for symbol,length in sorted(enumerate(table),key=lambda a:(a[1],a[0])):
  if not length:continue
  code<<=length-previous;trie[(length,code)]=symbol;code+=1;previous=length
 exp=np.empty(count,np.uint32);index=0;current=0;used=0
 for byte in payload:
  for shift in range(7,-1,-1):
   current=(current<<1)|((byte>>shift)&1);used+=1
   if (used,current) in trie:
    exp[index]=trie[used,current];index+=1;current=0;used=0
    if index==count:return (low&0x7fffff)|((low>>23)<<31)|(exp<<23)
 raise ValueError('truncated exponent stream')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='artifacts/column32-v1-prefix/queries');ap.add_argument('--output',default='artifacts/prefix-dense/capacity.json');a=ap.parse_args();source=ROOT/a.source;cache=(source/'cache.json').read_bytes();meta=json.loads(cache);cases=[]
 main=ROOT/'artifacts/output-pairs-v2-617';report=json.loads((main/'report.json').read_text());helper=ROOT/'artifacts/bounded_i16/native/release/quantized_input';dest=(ROOT/a.output).parent;dest.mkdir(parents=True,exist_ok=True)
 for layer in range(32):
  if (layer+1)%4==0:continue
  path=source/f'states/layer-{layer:02d}.npz'
  with np.load(path,allow_pickle=False) as f:state=f['delta'].copy()
  assert state.shape==(32,128,128) and state.dtype==np.float32 and np.isfinite(state).all()
  bits=state.astype('<f4').view('<u4');table,payload,raw,compressed=encode(bits)
  for mant,flag in [(raw,False),(compressed,True)]:assert decode(table,payload,mant,bits.size,flag).tobytes()==bits.tobytes()
  sizes=dict(huffman_exponent_raw_mantissa=len(table)+len(payload)+len(raw),huffman_exponent_zlib_mantissa=len(table)+len(payload)+len(compressed))
  # Conservative header plus q bytes/scales and both original rank64 A products.
  # This assumes these operands are produced by a PREVIOUS canister query;
  # no host computation or query packing has been implemented here.
  with np.load(path,allow_pickle=False) as f:conv=f['conv'].copy()
  assert np.all((conv.view(np.uint32)&65535)==0)
  query=next(q for q in report['queries'] if q['op']=='delta_full_log_integer' and f'.layers.{layer}.' in q['tensor'])
  request=main/'queries'/f'{query["index"]:06d}.request.bin';header,values=decode_frame(request.read_bytes());assert header['dims'][0]==87
  ip=dest/f'layer-{layer:02d}.input.bin';qp=dest/f'layer-{layer:02d}.q.bin';ip.write_bytes(values[:87*2560].astype('<f4').tobytes())
  subprocess.run([str(helper),'87','2560',str(ip),str(qp)],check=True)
  qbytes=qp.read_bytes();assert len(qbytes)==87*2560 and 128 not in qbytes
  compressed_q=zlib.compress(qbytes,1);assert zlib.decompress(compressed_q)==qbytes
  # Gate A/B are non-LoRA INT8 projections and can use the same prepared q.
  # conv is verified BF16; qkv/z rank64 A products and q scales remain F32.
  extra=16424+len(compressed_q)+87*10*4+2*87*64*4+3*8192*2
  head_sizes=[]
  for head in range(32):
   hb=bits[head];ht,hp,hm,hc=encode(hb)
   assert decode(ht,hp,hm,hb.size,False).tobytes()==hb.tobytes()
   head_sizes.append(len(ht)+len(hp)+len(hm)+8)
  log_path=ROOT/'artifacts/output-pairs-v2-prefix/queries/states'/f'layer-{layer:02d}.npz'
  with np.load(log_path,allow_pickle=False) as f:log=f['delta_log'].copy()
  assert log.shape==(45*6176,) and np.isfinite(log).all()
  k=log[:45*2048].reshape(45,16,128);assert np.all((k.view(np.uint32)&65535)==0)
  # Keep pairs together so each remaining pair has one exact BF16 K stream.
  pair_log_bytes=45*(128*2+2*128*4+2*4)
  order=sorted(range(16),key=lambda j:head_sizes[2*j]+head_sizes[2*j+1]-pair_log_bytes)
  budgets={}
  for label,other in [('original_bf16_input',16424+2*(87*2560+3*8192)),('prepared_compressed_input',extra)]:
   total=other+16*pair_log_bytes+4;chosen=[]
   for pair in order:
    increment=head_sizes[2*pair]+head_sizes[2*pair+1]-pair_log_bytes
    if total+increment<=2000000:total+=increment;chosen.extend([2*pair,2*pair+1])
   budgets[label]=dict(dense_heads=sorted(chosen),dense_head_count=len(chosen),remaining_log_heads=32-len(chosen),estimated_request_bytes=total,eliminated_prefix_outer_products_fraction=len(chosen)/32)
  cases.append(dict(layer=layer,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),quantized_bytes_sha256=hashlib.sha256(qbytes).hexdigest(),q_raw_bytes=len(qbytes),q_compressed_bytes=len(compressed_q),prefix_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),hybrid_capacity=budgets,state_bytes=bits.nbytes,encoded_bytes=sizes,estimated_request_bytes={k:v+extra for k,v in sizes.items()},fits_2000000={k:v+extra<=2000000 for k,v in sizes.items()},bitwise_roundtrip=True))
 result=dict(scope='Host-only exact state encoding and capacity probe; not a Wasm decoder, instruction, judgment or query-count measurement',source=a.source,cache_sha256=hashlib.sha256(cache).hexdigest(),model=meta['model'],pack_hash=meta['pack_hash'],suffix_tokens=87,probe_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),header_budget=16424,conv_budget='BF16 exact, verified from each saved state',quantizer_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),quantizer_source_sha256=hashlib.sha256((ROOT/'scripts/quantized_input.rs').read_bytes()).hexdigest(),native_runtime_sha256=hashlib.sha256((ROOT/'artifacts/bounded_i16/native/release/deps/libimajev_runtime-5a35000a9c0d4e61.rlib').read_bytes()).hexdigest(),cases=cases)
 p=ROOT/a.output;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(layers=len(cases),ranges={k:[min(c['estimated_request_bytes'][k] for c in cases),max(c['estimated_request_bytes'][k] for c in cases)] for k in sizes},fits={k:sum(c['fits_2000000'][k] for c in cases) for k in sizes},hybrid_dense_head_ranges={label:[min(len(c['hybrid_capacity'][label]['dense_heads']) for c in cases),max(len(c['hybrid_capacity'][label]['dense_heads']) for c in cases)] for label in budgets})))
if __name__=='__main__':main()
