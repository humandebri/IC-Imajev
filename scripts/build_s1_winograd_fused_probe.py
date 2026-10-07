#!/usr/bin/env python3
"""Fuse first-use Winograd coefficient initialization into the unchanged dot stream."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]


def weight(m,j,k):
    get=lambda x:f'local.get $w{x}_{j}_{k}'
    tee=lambda x:f'local.tee $w{x}_{j}_{k}'
    load=lambda plane:f'(v128.load8x8_s offset={k*8}(local.get $bp{plane}))'
    if m in [0,1,2]:return [load([0,1,3][m]),tee(m)]
    if m==3:
        return [get(2),load(2),get(0),'i16x8.sub',tee(4),'i16x8.sub',tee(5),get(1),'i16x8.sub',tee(3)]
    if m in [4,5]:return [get(m)]
    assert m==6
    return [get(5),get(0),'i16x8.sub',tee(6)]


def main():
    old=ROOT/'artifacts/s1-winograd-v1'
    report=json.loads((old/'build/report.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for key in ['source_hashes','dependency_hashes']:
        assert all(sha(ROOT/p)==h for p,h in report[key].items())
    p=ROOT/'scripts/build_s1_winograd_probe.py'
    source=p.read_text().replace('artifacts/s1-winograd-v1','artifacts/s1-winograd-fused-v1')
    assert source.count('D.mkdir(exist_ok=False);B=')==1
    source=source.replace('D.mkdir(exist_ok=False);B=','D.mkdir(exist_ok=True);B=')
    lo=source.index('   if first:\n    for k in range(32):')
    hi=source.index('\n   for m in range(7):\n    if j==0',lo)+1
    source=source[:lo]+'''   if first:
    for plane in range(4):out.append(f'(local.set $bp{plane}(i32.add(local.get $wp{plane})(i32.mul(local.get $cols)(i32.const {j}))))')
'''+source[hi:]
    anchor="     out+=[f'(local.tee $x{m}_{k}(v128.load offset={k*16}(local.get $qp)))'if j==0 else f'local.get $x{m}_{k}',f'local.get $w{m}_{j}_{k}','i32x4.dot_i16x8_s']"
    assert source.count(anchor)==1
    source=source.replace(anchor,"     out+=[f'(local.tee $x{m}_{k}(v128.load offset={k*16}(local.get $qp)))'if j==0 else f'local.get $x{m}_{k}']\n     out+=__import__('build_s1_winograd_fused_probe').weight(m,j,k) if first else [f'local.get $w{m}_{j}_{k}']\n     out+=['i32x4.dot_i16x8_s']")
    anchor=" lines.append('(local $value v128)')";assert source.count(anchor)==1
    source=source.replace(anchor,anchor+"\n for plane in range(4):lines.append(f'(local $bp{plane} i32)')")
    d=ROOT/'artifacts/s1-winograd-fused-v1';d.mkdir(exist_ok=False)
    (d/'frozen-builder.py').write_text(source)
    files=[Path(__file__),p,old/'build/report.json',d/'frozen-builder.py']
    (d/'entry-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):sha(f)for f in files},indent=2)+'\n')
    exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
