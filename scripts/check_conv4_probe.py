#!/usr/bin/env python3
"""Verify conv SIMD against unchanged runtime and independent ordered F32 oracle."""
import hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CANISTER='zm54s-at777-77775-aaa5q-cai';HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/conv4-simd-v3/check';d.mkdir(exist_ok=False);b=json.loads((d.parent/'build/report.json').read_text())
 for g in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[g].items())
 status=json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True));assert status['module_hash'].removeprefix('0x')==b['module']
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());assert m['model']==sha(ROOT/'MODEL_LOCK.json');t=next(t for t in m['tensors'] if t['name']=='model.language_model.layers.0.linear_attn.conv1d.weight')
 assert (t['rows'],t['cols'],t['dtype'],t['bytes'])==(8192,4,'int8',65536)
 with (ROOT/'checkpoints/full-int8.pack').open('rb') as f:f.seek(t['offset']);raw=f.read(t['bytes'])
 q=np.frombuffer(raw[:8192*4],dtype='i1').reshape(8192,4);sc=np.frombuffer(raw[8192*4:],dtype='<f4');weights=np.multiply(q.astype('<f4'),sc[:,None],dtype=np.float32)
 actual_path=ROOT/'artifacts/f32_block/check/617.input.bin';actual=np.frombuffer(actual_path.read_bytes(),dtype='<f4')
 pattern=np.array([0.,-0.,np.float32(2**-149),np.float32(-2**-149),1.,-1.,1.00390625,-1.01171875,0.75,-0.25],dtype='<f4')
 cases=[(f'boundary-{n}-{c}',n,c,np.resize(pattern,(n+3,c)),np.resize(np.array([1.,-1.,.5,-.25],dtype='<f4'),(c,4))) for n,c in [(1,4),(3,8),(8,12),(56,4)]]
 cases += [(f'weights-{n}-{c}',n,c,np.resize(actual,(n+3,c)),weights[:c]) for n,c in [(1,8192),(48,8192),(56,8192),(57,2560),(67,2560),(132,2560)]]
 rows=[]
 for label,n,c,x,w in cases:
  inp=d/(label+'.input.bin');inp.write_bytes(struct.pack('<III',n,c,4)+x.astype('<f4').tobytes()+w.astype('<f4').tobytes())
  y=np.zeros((n,c),dtype='<f4')
  for j in range(4):y=np.add(y,np.multiply(x[j:j+n],w[:,j],dtype=np.float32),dtype=np.float32)
  expected=y.astype('<f4').tobytes()+x[-3:].astype('<f4').tobytes();expected_hash=list(hashlib.sha256(expected).digest());exp=d/(label+'.preactivation.bin');exp.write_bytes(expected)
  measured=[]
  for method in range(4):
   arg=d/f'{label}-{method}.args.bin';subprocess.run([str(HELPER),'query',str(method),str(inp),str(arg)],check=True)
   rawreply=subprocess.check_output(['icp','canister','call',CANISTER,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--output','hex'],text=True)
   reply=d/f'{label}-{method}.hex';reply.write_text(rawreply);v=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True))
   assert v['output_values']==(n+3)*c and v['total_instructions']==v['input_prepare_instructions']+v['project_instructions']
   if method>=2:assert v['digest']==expected_hash,(label,method)
   measured.append(dict(measurement=v,reply_sha256=sha(reply)))
  assert measured[0]['measurement']['digest']==measured[1]['measurement']['digest'],label
  row=dict(label=label,n=n,c=c,input_sha256=sha(inp),preactivation_sha256=sha(exp),measurements=measured,bitwise_equal=True,reduction_percent=100*(1-measured[1]['measurement']['project_instructions']/measured[0]['measurement']['project_instructions']))
  rows.append(row);print(json.dumps(dict(label=label,reduction_percent=row['reduction_percent'])),flush=True)
 r=dict(module=b['module'],cases=rows,all_bits_equal=True,source_sha256=sha(Path(__file__)),helper_sha256=sha(HELPER),weight_tensor=t,packed_weights_sha256=hashlib.sha256(raw).hexdigest(),actual_values_source_sha256=sha(actual_path),scope='Isolated conv_state with real dequantized layer0 weights and saved activation values as operands; not an exported convolution intermediate or full inference. Original scalar runtime output agrees and preactivation has an independent ordered F32 oracle.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
