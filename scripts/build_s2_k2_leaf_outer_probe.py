#!/usr/bin/env python3
"""Leaf-outer tile96 diagnostic against the fully verified current guard-fold K2."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 old=ROOT/'scripts/build_s2_k2_probe.py';d=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1';d.mkdir(exist_ok=False)
 s=old.read_text().replace('s2-k2-probe-v1','s2-k2-leaf-outer-probe-v1').replace('s2-k2-kernels-v1','s2-k2-leaf-outer-kernels-v1').replace('update-k2-compact-v1','update-k2-pair-guard-fold-v1').replace('d.mkdir(exist_ok=False)','d.mkdir(exist_ok=True)')
 for before,after in [('rows-r>=80{80}','rows-r>=96{96}'),('(80,true)','(96,true)'),('(80,false)','(96,false)'),('tile80_seed','tile96_seed'),('tile80','tile96'),('output_tile=80','output_tile=96')]:
  assert before in s,before;s=s.replace(before,after)
 s=s.replace('Benchmark rank49 compact K2 direct outputs against actual current full K2 runtime.','Benchmark leaf-outer rank49 tile96 against current guard-fold K2; no performance assumed.')
 p=d/'frozen-builder.py';p.write_text(s);(d/'entry-builder-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),old,p]},indent=2)+'\n');exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
