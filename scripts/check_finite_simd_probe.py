#!/usr/bin/env python3
"""Integer-bit oracle for all BF16 patterns, vector tails, and nonfinite placements."""
from pathlib import Path
import hashlib,json,struct,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/finite-simd-v1/check';B=D.parent/'build'
TARGET='zm54s-at777-77775-aaa5q-cai';HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle(bits,stride):
 return [int(np.all((bits[i:i+stride]&0x7f800000)!=0x7f800000))for i in range(0,len(bits),stride)]or[1]
def main():
 D.mkdir(exist_ok=False);build=json.loads((B/'report.json').read_text())
 status=json.loads(subprocess.check_output(['icp','canister','status',TARGET,'--network','local','--identity','imajev-local','--json'],text=True));assert status['status']=='Running' and status['module_hash'].removeprefix('0x')==build['module']
 did=D/'probe.did';did.write_text('service:{project:(vec nat8,nat8)->(record{digest:vec nat8;quantize_instructions:nat64;input_prepare_instructions:nat64;project_instructions:nat64;total_instructions:nat64;output_values:nat64;heap_pages:nat64}) query;}')
 hashes={**build['source_hashes'],**build['dependency_hashes'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(HELPER.relative_to(ROOT)):sha(HELPER),str(did.relative_to(ROOT)):sha(did)}
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 cases=[];pattern=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff],dtype='<u4')
 for n in list(range(0,18))+[31,32,33,63,64,65,127,128,129,255,256,257]:cases.append((f'tail-{n}',np.resize(pattern,n),max(n,1),'finite tails'))
 for n in [48*2560,56*2560,57*2560,48*8192]:cases.append((f'finite-{n}',np.resize(pattern,n),n,'full finite scan'))
 real=np.fromfile(ROOT/'artifacts/f32_block/check/617.input.bin',dtype='<u4');assert len(real)<=450000;cases.append(('real-617',real,len(real),'saved real F32 activation'))
 # Every stored BF16 bit pattern in each SIMD4 vector of a SIMD16 block.
 for position in [0,4,8,12]:
  for begin in range(0,65536,20000):
   count=min(20000,65536-begin);values=np.zeros((count,16),dtype='<u4');values[:,position]=np.arange(begin,begin+count,dtype='<u4')<<16
   cases.append((f'bf16-{position}-{begin}',values.ravel(),16,'all65536 BF16 patterns in each SIMD16 vector'))
 unusual=[0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00000,0xffc00000,0x7fffffff,0xffffffff]
 values=[]
 for bits in unusual:
  for pos in range(65):
   row=np.resize(pattern,65);row[pos]=bits;values.append(row)
 cases.append(('all-invalid-positions',np.array(values,dtype='<u4').ravel(),65,'each Inf/signaling/quiet NaN at all64 positions plus scalar tail'))
 cases.append(('random-bits',np.random.default_rng(77217).integers(0,2**32,size=128*2048,dtype=np.uint32),128,'independent arbitrary F32 bits'))
 result=[]
 for label,bits,stride,scope in cases:
  p=D/f'{label}.input.bin';p.write_bytes(struct.pack('<I',stride)+bits.astype('<u4').tobytes());assert p.stat().st_size<=1900000
  expected=oracle(bits,stride);measured=[]
  for method in range(4):
   arg=D/f'{label}-{method}.args.bin';reply=D/f'{label}-{method}.hex';subprocess.run([str(HELPER),'query',str(method),str(p),str(arg)],check=True)
   start=time.monotonic();raw=subprocess.check_output(['icp','canister','call',TARGET,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--candid',str(did),'--output','hex'],text=True);reply.write_text(raw)
   m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));assert m['digest']==expected,(label,method);assert m['output_values']==len(bits);assert m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
   measured.append(dict(method=method,measurement=m,reply_sha256=sha(reply),seconds=time.monotonic()-start))
  row=dict(label=label,values=len(bits),stride=stride,scope=scope,input_sha256=sha(p),measurements=measured,all_predicates_equal=True);result.append(row)
  if label.startswith('finite-')or label=='real-617':print(json.dumps(dict(label=label,body=[x['measurement']['project_instructions']for x in measured],reductions=[100*(1-x['measurement']['project_instructions']/measured[0]['measurement']['project_instructions'])for x in measured[1:]])),flush=True)
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 r=dict(complete=True,module=build['module'],cases=result,queries=len(result)*4,all_predicates_equal=True,source_hashes=hashes,scope='Finite predicate only. Input decode and reply encoding excluded equally from body cost; no inference performance claim.')
 (D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(cases=len(result),queries=r['queries'],all_predicates_equal=True)))
if __name__=='__main__':main()
