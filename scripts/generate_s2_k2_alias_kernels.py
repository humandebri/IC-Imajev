#!/usr/bin/env python3
"""Write rank49 B DAG leaves directly to persistent coefficient slots."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/generate_s2_k2_kernels.py';d=ROOT/'artifacts/s2-k2-alias-kernels-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('s2-k2-kernels-v1','s2-k2-alias-kernels-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 anchor=" symbol=f'__imajev_s2_k2_{tile}'"
 patch=" leafmap={name:m for m,(_,name)in enumerate(leaves)};assert len(leafmap)==49\n def cached(name,j,k):return f'w{leafmap[name]}_{j}_{k}'if name in leafmap else name\n"
 assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 s=s.replace("for i in range(16):lines.append(f'(local $wp{i} i32)(local $bp{i} i32)(local $b{i} v128)')","for i in range(16):\n  lines.append(f'(local $wp{i} i32)(local $bp{i} i32)')\n  if f'b{i}'not in leafmap:lines.append(f'(local $b{i} v128)')")
 s=s.replace("for name,_ in b.nodes:lines.append(f'(local ${name} v128)')","for name,_ in b.nodes:\n  if name not in leafmap:lines.append(f'(local ${name} v128)')")
 before="     for bi in range(16):out.append(f'(local.set $b{bi}(v128.load8x8_s offset={k*8}(local.get $bp{bi})))')\n     for name,terms in b.nodes:out+=ns['expression'](terms,lambda n:f'local.get ${n}','i16x8')+[f'local.set ${name}']\n     for m,(_,name)in enumerate(leaves):out+=[f'local.get ${name}',f'local.set $w{m}_{j}_{k}']"
 after="     for bi in range(16):out.append(f'(local.set ${cached(f\"b{bi}\",j,k)}(v128.load8x8_s offset={k*8}(local.get $bp{bi})))')\n     for name,terms in b.nodes:out+=ns['expression'](terms,lambda n:f'local.get ${cached(n,j,k)}','i16x8')+[f'local.set ${cached(name,j,k)}']"
 assert s.count(before)==1;s=s.replace(before,after)
 (d/'frozen-generator.py').write_text(s)
 files=[Path(__file__),old,d/'frozen-generator.py'];(d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-generator.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
