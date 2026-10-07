#!/usr/bin/env python3
"""Verify SIMD finalization against scalar Wasm and independent F32/BF16 arithmetic."""
import hashlib,json,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CANISTER='zm54s-at777-77775-aaa5q-cai'
HELPER=ROOT/'artifacts/f32-block-native/release/f32_args'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def bf(v):
 b=struct.unpack('<I',struct.pack('<f',v))[0]
 return struct.unpack('<f',struct.pack('<I',((b+0x7fff+((b>>16)&1))&0xffffffff)&0xffff0000))[0]
def main():
 d=ROOT/'artifacts/lora-finish-v1/check';d.mkdir(exist_ok=False)
 build=json.loads((d.parent/'build/report.json').read_text())
 status=json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True));assert status['module_hash'].removeprefix('0x')==build['module']
 for g in ['source_hashes','dependency_hashes']:
  assert all(sha(ROOT/p)==h for p,h in build[g].items())
 actual=(ROOT/'artifacts/f32_block/check/617.input.bin').read_bytes();actual=list(struct.unpack('<'+str(len(actual)//4)+'f',actual))
 bits=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f808000,0xbf818000,0x3f807fff,0xbf817fff,0x3f808001,0xbf818001,0x3e123456,0xbe123456]
 pattern=[struct.unpack('<f',struct.pack('<I',b))[0] for b in bits]
 cases=[('boundary-'+str(n),[pattern[i%len(pattern)] for i in range(n)], [pattern[(i*3+1)%len(pattern)] for i in range(n)],scale) for n,scale in [(1,1.25),(3,-1.25),(4,0.0),(5,-0.0),(7,1.25),(4096,1.25),(24576,-1.25),(196608,1.25)]]
 cases += [('real-617',actual,actual[::-1],1.25)]
 result=[]
 for label,base,z,scale in cases:
  data=struct.pack('<f',scale)+struct.pack('<'+str(len(base))+'f',*base)+struct.pack('<'+str(len(z))+'f',*z)
  inp=d/(label+'.input.bin');inp.write_bytes(data)
  expected=b''.join(struct.pack('<f',bf(f32(bf(v)+bf(f32(scale*w))))) for v,w in zip(base,z));exp=d/(label+'.expected.bin');exp.write_bytes(expected)
  measurements=[]
  for method in [0,1]:
   arg=d/f'{label}-{method}.args.bin';subprocess.run([str(HELPER),'query',str(method),str(inp),str(arg)],check=True)
   raw=subprocess.check_output(['icp','canister','call',CANISTER,'project','--args-file',str(arg),'--args-format','bin','--query','--network','local','--identity','imajev-local','--output','hex'],text=True)
   reply=d/f'{label}-{method}.hex';reply.write_text(raw)
   m=json.loads(subprocess.check_output([str(HELPER),'decode',str(reply),'measurement'],text=True));assert bytes(m.pop('digest'))==expected,label
   assert m['output_values']==len(base) and m['total_instructions']==m['input_prepare_instructions']+m['project_instructions']
   measurements.append(dict(measurement=m,reply_sha256=sha(reply)))
  row=dict(label=label,values=len(base),scale=scale,input_sha256=sha(inp),expected_sha256=sha(exp),measurements=measurements,bitwise_equal=True,reduction_percent=100*(1-measurements[1]['measurement']['project_instructions']/measurements[0]['measurement']['project_instructions']))
  result.append(row);print(json.dumps(dict(label=label,reduction_percent=row['reduction_percent'])),flush=True)
 r=dict(module=build['module'],cases=result,helper_sha256=sha(HELPER),source_sha256=sha(Path(__file__)),all_bits_equal=True,scope='Isolated finalization; actual input values are used as arithmetic operands, not a full inference oracle.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
