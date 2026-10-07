#!/usr/bin/env python3
"""Raw output-lane permutation with same original weights and current control."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s2-k2-leaf-outer-probe-v1/frozen-builder.py';d=ROOT/'artifacts/s2-contiguous-roots-probe-v1';d.mkdir(exist_ok=False)
 kernels=ROOT/'artifacts/s2-contiguous-roots-kernels-v1';plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';(kernels/'plan.py').write_bytes(plan.read_bytes())
 source=original.read_text().replace('s2-k2-leaf-outer-probe-v1','s2-contiguous-roots-probe-v1').replace('s2-k2-leaf-outer-kernels-v1','s2-contiguous-roots-kernels-v1')
 source=source.replace("audit=json.loads((kernels/'layout-audit.json').read_text());assert audit['complete']and audit['all_integer_roots_equal']","audit=json.loads((kernels/'execution-report.json').read_text());assert audit['complete']and audit['conditions']==104;validation=json.loads((kernels/'wasm-validation.json').read_text());assert validation['all4_wasm_validated'];assert all(sha(ROOT/p)==h for r in [audit,validation] for p,h in r['source_hashes'].items())")
 before=" assert base.count('for lane in 0..4{data[')==1;";assert source.count(before)==1
 source=source.replace(before," base=base.replace('(group*16+lane*4+j%4)*cols','(group*16+(j%4)*4+lane)*cols').replace('let r=row*4+ri;','let r=(row/4)*16+ri*4+row%4;')\n"+before)
 source=source.replace("kernels/'layout-audit.json'","kernels/'execution-report.json'")
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen,plan,kernels/'plan.py']},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
