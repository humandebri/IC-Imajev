#!/usr/bin/env python3
"""WWC streaming integer DAG, batch88/tile256, one multi-value leaf helper."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=ROOT/'artifacts/s3-streaming-shared-kernels-v1';d.mkdir(exist_ok=False)
 source=ROOT/'artifacts/rank343-streaming-reconstruction-v1/report.json';r=json.loads(source.read_text())
 assert r['complete'] and r['register_capacity']==101
 for p,h in r['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 T=11;J=8;params=['q','w','cols','start','sx','stride','sw','sums','n']
 lines=['(module (memory 1)', '(func(export "__imajev_s3_streaming_256")'+''.join(f'(param ${p} i32)'for p in params)]
 lines+=['(local $t i32)(local $yp i32)(local $input_stride i32)(local $output_bytes i32)']
 for ti in range(T):
  for j in range(J):
   for slot in range(101):lines.append(f'(local $p{ti}_{j}_{slot} v128)')
 for row in range(T*8):lines.append(f'(local $sx{row} v128)')
 for i in range(J*8):lines.append(f'(local $sw{i} v128)')
 for i in range(J*8):lines.append(f'(local.set $sw{i}(v128.load offset={i*16}(local.get $sw)))')
 lines+=['(local.set $input_stride(i32.shr_u(local.get $cols)(i32.const 8)))','(local.set $output_bytes(i32.shl(local.get $stride)(i32.const 2)))','(block $done(loop $batch','(br_if $done(i32.ge_u(local.get $t)(local.get $n)))']
 for row in range(T*8):lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {row}))(local.get $n))(then(local.set $sx{row}(v128.load32_splat(i32.add(local.get $sx)(i32.shl(i32.mul(i32.add(local.get $t)(i32.const {row}))(local.get $input_stride))(i32.const 2)))))))')
 for event in r['events']:
  if event['kind']=='leaf':
   lines+=['local.get $q','local.get $w','local.get $cols','local.get $start','local.get $t',f'i32.const {event["index"]}','local.get $n','i32.const 0','i32.const 0','call $leaf']
   for ti in reversed(range(T)):
    for j in reversed(range(J)):lines.append(f'local.set $p{ti}_{j}_{event["destination"]}')
  else:
   for ti in range(T):
    lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti*8}))(local.get $n))(then')
    for j in range(J):
     for k,operand in enumerate(event['operands']):
      if k==0 and operand['sign']<0:lines.append('v128.const i32x4 0 0 0 0')
      lines.append(f'local.get $p{ti}_{j}_{operand["slot"]}')
      if k or operand['sign']<0:lines.append('i32x4.add'if operand['sign']>0 else'i32x4.sub')
     lines.append(f'local.set $p{ti}_{j}_{event["destination"]}')
    lines.append('))')
 def shuffle(mask):return'i8x16.shuffle '+' '.join(str(i*4+b)for i in mask for b in range(4))
 for ti in range(T):
  for row in range(8):
   absolute=ti*8+row
   lines += [f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {absolute}))(local.get $n))(then',f'(local.set $yp(i32.add(local.get $sums)(i32.mul(i32.add(local.get $t)(i32.const {absolute}))(local.get $output_bytes))))']
   for j in range(J):
    for lane in range(4):
     for base in [0,4]:
      offset=j*128+(lane*8+base)*4;mask=[0,4,1,5]if lane<2 else[2,6,3,7]
      lines+=['local.get $yp','local.get $yp',f'v128.load offset={offset}']
      for pair in [0,2]:lines +=[f'local.get $p{ti}_{j}_{r["root_registers"][row][base+pair]}',f'local.get $p{ti}_{j}_{r["root_registers"][row][base+pair+1]}',shuffle(mask)]
      lines +=[shuffle([0,1,4,5]if lane%2==0 else[2,3,6,7]),'f32x4.convert_i32x4_s',f'local.get $sx{absolute}','f32x4.mul',f'local.get $sw{j*8+lane*2+base//4}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
   lines.append('))')
 lines+=['(local.set $t(i32.add(local.get $t)(i32.const 88)))','br $batch','))',')']
 main_locals=sum(x.count('(local $')for x in lines);assert main_locals<10000
 lines+=['(func $leaf'+''.join(f'(param ${p} i32)'for p in ['q','w','cols','start','t','m','n','unused0','unused1'])+'(result '+' '.join(['v128']*(T*J))+')','(local $qp i32)(local $qoff i32)']
 for j in range(J):
  lines.append(f'(local $bp{j} i32)')
  for k in range(16):lines.append(f'(local $w{j}_{k} v128)')
 for k in range(16):lines.append(f'(local $x{k} v128)')
 for ti in range(T):
  for j in range(J):lines.append(f'(local $r{ti}_{j} v128)')
 for j in range(J):
  lines.append(f'(local.set $bp{j}(i32.add(i32.add(i32.load offset={j*4}(local.get $w))(i32.mul(i32.shr_u(local.get $start)(i32.const 8))(i32.const 87808)))(i32.shl(local.get $m)(i32.const 8))))')
  for k in range(16):lines.append(f'(local.set $w{j}_{k}(v128.load offset={k*16}(local.get $bp{j})))')
 for ti in range(T):
  lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti*8}))(local.get $n))(then')
  lines.append(f'(local.set $qoff(i32.add(i32.shr_u(i32.mul(i32.add(local.get $t)(i32.const {ti*8}))(local.get $cols))(i32.const 5))(i32.shr_u(local.get $start)(i32.const 2))))')
  lines.append('(local.set $qp(i32.add(i32.load(i32.add(local.get $q)(i32.shl(local.get $m)(i32.const 2))))(local.get $qoff)))')
  for j in range(J):
   for k in range(16):
    lines.append(f'(local.tee $x{k}(v128.load32_splat offset={k*4}(local.get $qp)))'if j==0 else f'local.get $x{k}')
    lines +=[f'local.get $w{j}_{k}','i32x4.dot_i16x8_s']
    if k:lines.append('i32x4.add')
   lines.append(f'local.set $r{ti}_{j}')
  lines.append('))')
 for ti in range(T):
  for j in range(J):lines.append(f'local.get $r{ti}_{j}')
 lines+=['))'];text='\n'.join(lines)+'\n';assert text.count('(')==text.count(')')
 p=d/'256.wat';p.write_text(text)
 subprocess.run([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-audit'),'wat',str(p),str(d/'256.wasm')],check=True)
 files=[Path(__file__),source,p,d/'256.wasm'];report=dict(token_batch=88,output_tile=256,main_locals=main_locals,helper_results=88,leaf_calls=343,streaming_events=len(r['events']),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},wasm_execution_verified=False,performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
