#!/usr/bin/env python3
"""Permute raw output lanes so rank49 roots store four adjacent original outputs."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-contiguous-roots-kernels-v1';d.mkdir(exist_ok=False)
 original=ROOT/'scripts/generate_s2_k2_leaf_outer_kernels.py';source=original.read_text()
 begin=source.index('    for ri in range(4):out+=');end=source.index("    out+=['))']",begin)
 source=source[:begin]+'''    for part in range(4):
     offset=j*64+part*16
     out+=['local.get $yp']+(['v128.const i32x4 0 0 0 0']if seed else ['local.get $yp',f'v128.load offset={offset}'])+[f'local.get ${register(regs[ti][part],j)}','f32x4.convert_i32x4_s',f'local.get $sx{ti}','f32x4.mul',f'local.get $sw{j*4+part}','f32x4.mul','f32x4.add',f'v128.store offset={offset}']
'''+source[end:]
 frozen=d/'frozen-generator.py';frozen.write_text(source);gen={'__name__':'generator','__file__':str(original)};exec(compile(source,str(frozen),'exec'),gen)
 plan=ROOT/'artifacts/s2-k2-leaf-outer-kernels-v1/plan.py';ns={'__name__':'plan','__file__':str(plan)};exec(compile(plan.read_text(),str(plan),'exec'),ns)
 files=[Path(__file__),original,frozen,plan];items=[]
 for tile in [96,16]:
  for seed in [False,True]:
   text,count,symbol=gen['kernel'](ns,tile,seed);assert 'i8x16.shuffle'not in text
   p=d/(str(tile)+('_seed'if seed else '')+'.wat');p.write_text(text);files.append(p);items.append(dict(path=str(p.relative_to(ROOT)),tile=tile,seed=seed,symbol=symbol,locals=count))
 report=dict(rank=49,kernels=items,raw_capacity_unchanged=True,output_mapping='logical_Ncolumn*4+SIMDlane, instead of SIMDlane*4+logical_Ncolumn',all_output_shuffles_removed=True,integer_DAG_and_dot_order_unchanged=True,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},wasm_execution_verified=False,performance_verified=False,scope=__doc__)
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(items))
if __name__=='__main__':main()
