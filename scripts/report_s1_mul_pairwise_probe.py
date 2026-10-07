#!/usr/bin/env python3
"""Re-decode bounded multiply/pairwise measurements and compare to previous exact adaptive kernel."""
from pathlib import Path
import ast,hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-mul-pairwise-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=json.loads((D/'build/report.json').read_text());r=json.loads((D/'check/report.json').read_text());assert b['wasm_sha256']==r['wasm_sha256']==sha(D/'build/diagnostic.wasm');assert len(r['cases'])==21 and r['ordinary_queries']==42
 for manifest in [b['source_hashes'],b['dependency_hashes'],r['source_hashes']]:assert all(sha(ROOT/p)==h for p,h in manifest.items())
 symbolic=ROOT/'scripts/generate_wat_s1.py';tree=ast.parse(symbolic.read_text());nodes=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ['A','B','C']for t in n.targets)];ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(symbolic),'exec'),ns)
 proof=json.loads((D/'integer-bounds.json').read_text());assert proof['changed_products']==[1,2,3,4]
 for m in proof['changed_products']:
  bound=sum(abs(v)for v in ns['A'][m].values())*127*sum(abs(v)for v in ns['B'][m].values())*128;assert proof['i16_product_absolute_bounds'][str(m)]==bound<32768
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';prev=ROOT/'artifacts/s1-adaptive160-v1/summary.json';by={v['label']:v for v in json.loads(prev.read_text())['cases']};cases=[]
 for c in r['cases']:
  assert sha(D/'check'/f"{c['label']}.input.bin")==c['input_sha256']
  for name in ['s1_pair_bounds','mul_pairwise']:
   m=c['measurements'][name];reply=ROOT/m['reply'];assert sha(reply)==m['reply_sha256'];raw=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True));assert all(m[k]==v for k,v in raw.items());assert raw['digest']==c['native']['digest'];assert raw['total_instructions']==sum(raw[k]for k in ['quantize_instructions','input_prepare_instructions','project_instructions'])
  before=c['measurements']['s1_pair_bounds']['total_instructions'];after=c['measurements']['mul_pairwise']['total_instructions'];assert abs(c['reduction_percent']-100*(1-after/before))<1e-10;row=dict(label=c['label'],tokens=c['tokens'],before=before,after=after,versus_control_gain_percent=c['reduction_percent'])
  if c['label']in by:old=by[c['label']];assert old['tokens']==c['tokens'];row.update(previous_adaptive=old['after'],versus_original_dot_gain_percent=100*(1-after/old['after']))
  cases.append(row)
 paths=[Path(__file__),ROOT/'scripts/build_s1_mul_pairwise_probe.py',ROOT/'scripts/check_s1_mul_pairwise_probe.py',D/'integer-bounds.json',D/'frozen-check.py',D/'check-upstream.sha256',D/'build/report.json',D/'check/report.json',symbolic,prev]+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'));paths=list(dict.fromkeys(paths));summary=dict(module=r['wasm_sha256'],all_saved_replies_verified=True,integer_bounds_verified=True,native_bits_equal=True,cases=cases,queries=42,whole_inference_verified=False,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in paths},scope='Isolated Q projection; unchanged weights/quantization/block order. Four bounded rank products use I16 mul plus widening pairwise add. Compare actual IC handler instruction counters, not assumed instruction pricing.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print('21 cases /42 queries, full output digests and I16 product bounds verified')
if __name__=='__main__':main()
