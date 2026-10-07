#!/usr/bin/env python3
"""Reduce streaming reconstruction using verified readiness-gap4 inlining."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/generate_s3_streaming_shared_kernel.py'
 d=ROOT/'artifacts/s3-selective-streaming-kernels-v1';d.mkdir(exist_ok=False)
 source=original.read_text().replace('s3-streaming-shared-kernels-v1','s3-selective-streaming-kernels-v1').replace('rank343-streaming-reconstruction-v1/report.json','rank343-selective-streaming-inline-v1/gap-4/report.json').replace("r['register_capacity']==101","r['register_capacity']==100").replace('range(101)','range(100)').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 begin=source.index('  else:\n   for ti in range(T):');end=source.index(' def shuffle(mask):',begin)
 source=source[:begin]+'''  else:
   for ti in range(T):
    lines.append(f'(if(i32.lt_u(i32.add(local.get $t)(i32.const {ti*8}))(local.get $n))(then')
    for j in range(J):
     for op,arg in event['instructions']:
      if op=='get':lines.append(f'local.get $p{ti}_{j}_{arg}')
      elif op=='zero':lines.append('v128.const i32x4 0 0 0 0')
      else:lines.append('i32x4.'+op)
     lines.append(f'local.set $p{ti}_{j}_{event["destination"]}')
    lines.append('))')
'''+source[end:]
 frozen=d/'frozen-generator.py';frozen.write_text(source)
 (d/'generator-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
