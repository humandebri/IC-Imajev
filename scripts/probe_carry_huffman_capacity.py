#!/usr/bin/env python3
"""Exact byte-capacity estimate for independent carry byte planes.

Uses canister-produced partial MLP state and prefix state. Does not implement a
Wasm decoder, estimate decoder instructions, or prove fused-query feasibility.
"""
import argparse,collections,hashlib,heapq,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',default='artifacts/prefix_codec/mlp-delta-fusion-check-v2');ap.add_argument('--prefix',default='artifacts/prefix_codec/full-state-layout-proof/prefix');ap.add_argument('--output',default='artifacts/prefix_codec/carry-huffman-capacity.json');a=ap.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest();cases=[]
def sizes(raw):
 count=collections.Counter(raw);heap=[(n,symbol,symbol)for symbol,n in count.items()];heapq.heapify(heap);serial=256
 if len(heap)==1:length={heap[0][2]:1}
 else:
  while len(heap)>1:
   left=heapq.heappop(heap);right=heapq.heappop(heap);heapq.heappush(heap,(left[0]+right[0],serial,(left[2],right[2])));serial+=1
  length={};stack=[(heap[0][2],0)]
  while stack:
   node,depth=stack.pop()
   if isinstance(node,int):length[node]=depth
   else:stack.extend((child,depth+1)for child in node)
 bits=sum(count[s]*l for s,l in length.items());packed=(bits+7)//8
 # Descriptor is mode:u8 + packed length:u32. Huffman adds 256 code lengths;
 # a constant plane needs only its symbol. All original bytes remain exact.
 mode='constant'if len(count)==1 else('huffman'if max(length.values())<=24 and packed+256<len(raw)else'raw')
 size=6 if mode=='constant'else(5+256+packed if mode=='huffman'else 5+len(raw))
 return dict(raw_bytes=len(raw),mode=mode,bytes=size,max_code_bits=max(length.values()),huffman_payload_bytes=packed)
for layer in [0,1,3]:
 source=ROOT/a.source/f'{layer:02d}-prepare-partial.response.bin';raw=source.read_bytes();h,v=decode(raw);n=h['dims'][0];c=n*2560;q=n*9216
 sp=ROOT/a.prefix/'queries/states'/f'layer-{layer+1:02d}.npz'
 with np.load(sp,allow_pickle=False)as state:conv=state['conv'].ravel();log=state['delta_log'].ravel()
 groups=[((v[:c].view('<u4')>>16).astype('<u2').tobytes(),2), (v[c:c+q].astype(np.int8).tobytes(),1), (v[c+q:].astype('<f4').tobytes(),4), ((conv.view('<u4')>>16).astype('<u2').tobytes(),2), ((log[:45*2048].view('<u4')>>16).astype('<u2').tobytes(),2), (log[45*2048:].astype('<f4').tobytes(),4)]
 planes=[]
 for group,(raw,width)in enumerate(groups):
  data=np.frombuffer(raw,dtype=np.uint8).reshape(-1,width)
  for plane in range(width):planes.append(dict(group=group,plane=plane,**sizes(data[:,plane].tobytes())))
 # Conservative maximum bounded JSON header plus frame/tag/digest metadata.
 total=sum(p['bytes']for p in planes)+16384+4+32+1
 cases.append(dict(layer=layer,tokens=n,prefix=45,partial_rows=256,source_reply_sha256=sha(source.read_bytes()),prefix_state_sha256=sha(sp.read_bytes()),planes=planes,estimated_frame_upper_bytes=total))
result=dict(scope=__doc__,script_sha256=sha(pathlib.Path(__file__).read_bytes()),cases=cases)
path=ROOT/a.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([(c['layer'],c['estimated_frame_upper_bytes'])for c in cases]))
