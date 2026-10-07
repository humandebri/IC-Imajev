#!/usr/bin/env python3
"""Dispatch once per token quartet, prune zero leaves statically in each tail path."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'artifacts/s2-k2-alias-kernels-v1/frozen-generator.py';d=ROOT/'artifacts/s2-k2-tail-dispatch-kernels-v1';d.mkdir(exist_ok=False)
 s=source.read_text().replace('s2-k2-alias-kernels-v1','s2-k2-tail-dispatch-kernels-v1')
 assert s.count(' def row(first):')==1;s=s.replace(' def row(first):',' def row(first,valid=4):')
 anchor="    if j==0:out.append(f'(local.set $qp(i32.add(i32.load offset={m*4}(local.get $q))(local.get $qoff)))')"
 s=s.replace(anchor,"    if min(i//4 for i in a.symbols[leaves[m][0]])>=valid:\n     out.append(f'(local.set $pc{m}(v128.const i32x4 0 0 0 0))');continue\n"+anchor)
 anchor=" lines+=['(block $done'"
 dispatch=" def dispatch(first):\n  out=[]\n  for valid in [4,3,2]:\n   out.append(f'(if(i32.ge_u(i32.sub(local.get $n)(local.get $t))(i32.const {valid}))(then');out+=row(first,valid);out.append(')(else')\n  out+=row(first,1);out+=['))']*3;return out\n"
 assert s.count(anchor)==1;s=s.replace(anchor,dispatch+anchor).replace(']+row(True)+',']+dispatch(True)+').replace(']+row(False)+',']+dispatch(False)+')
 (d/'frozen-generator.py').write_text(s);(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),source,d/'frozen-generator.py']},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-generator.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
