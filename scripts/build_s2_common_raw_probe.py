#!/usr/bin/env python3
"""Measure rank49 on shared four-plane raw format against unchanged current control."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s2-contiguous-roots-probe-v1/frozen-builder.py';d=ROOT/'artifacts/s2-common-raw-probe-v1';d.mkdir(exist_ok=True)
 kernels=ROOT/'artifacts/s2-common-raw-kernels-v1';plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';(kernels/'plan.py').write_bytes(plan.read_bytes())
 s=original.read_text().replace('s2-contiguous-roots-probe-v1','s2-common-raw-probe-v1').replace('s2-contiguous-roots-kernels-v1','s2-common-raw-kernels-v1')
 # Original frozen builder has canonical rank49 raw producer text; replace after its root-permutation rewrite.
 before=" assert base.count('for lane in 0..4{data[')==1;"
 assert s.count(before)==1
 insertion=""" base=base.replace('let stride=cols/4;let plane=(rows/4)*stride;', 'let plane=rows*cols/4;')
 oldpack='for group in 0..rows/16{for block in 0..cols/256{for k in 0..64{for j in 0..16{for lane in 0..4{data[j*plane+group*cols+block*256+(k/2)*8+lane*2+k%2]=w[(group*16+(j%4)*4+lane)*cols+block*256+j/4*64+k];}}}}}'
 newpack='for r in 0..rows{for c in 0..cols{let i=((c%256)/128*2+(r%8)/4)*plane+(r/8)*cols*2+(c/256)*512+((c%128)/2)*8+(r%4)*2+c%2;data[i]=w[r*cols+c];}}'
 assert oldpack in base;base=base.replace(oldpack,newpack)
 base=base.replace('let plane=self.rows*cols/16;', 'let plane=self.rows*cols/4;')
 oldwp='self.data.as_ptr().add(m*plane+(r/16)*cols)';newwp='self.data.as_ptr().add(((m/4/2)*2+(m%4)%2)*plane+(r/8+(m%4)/2)*cols*2+((m/4)%2)*256)';assert oldwp in base;base=base.replace(oldwp,newwp)
 oldhost='j*(self.rows*cols/16)+(row/4)*cols+block*256+(k/2)*8+(row%4)*2+k%2';newhost='((j/4/2)*2+(j%4)%2)*(self.rows*cols/4)+((row/4)*2+(j%4)/2)*cols*2+block*512+((j/4)%2)*256+(k/2)*8+(row%4)*2+k%2';assert oldhost in base;base=base.replace(oldhost,newhost)
"""
 s=s.replace(before,insertion+before.replace("==1","==0"))
 s=s.replace('src.mkdir(parents=True)','src.mkdir(parents=True,exist_ok=True)')
 # Node proof covers all104 conditions; Rust body validation follows in this build.
 a="validation=json.loads((kernels/'wasm-validation.json').read_text());assert validation['all4_wasm_validated'];assert all(sha(ROOT/p)==h for r in [audit,validation] for p,h in r['source_hashes'].items())"
 assert a in s;s=s.replace(a,"assert all(sha(ROOT/p)==h for p,h in audit['source_hashes'].items())")
 frozen=d/'frozen-builder.py';frozen.write_text(s);(d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen,plan,kernels/'plan.py']},indent=2)+'\n')
 exec(compile(s,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
