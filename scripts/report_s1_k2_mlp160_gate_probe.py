#!/usr/bin/env python3
"""Independently re-decode MLP gate diagnostic replies and archive evidence."""
from pathlib import Path
import json,hashlib,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s1-k2-mlp160-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=D/'check-gate/report.json';r=json.loads(p.read_text());b=json.loads((D/'build/report.json').read_text())
 assert len(r['cases'])==16 and r['ordinary_queries']==32
 for k in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[k].items())
 assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 assert r['wasm_sha256']==b['wasm_sha256']==sha(D/'build/diagnostic.wasm')
 helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args';cases=[]
 for c in r['cases']:
  assert c['rows']==9216 and c['cols']==2560
  assert sha(D/'check-gate'/f"{c['label']}.input.bin")==c['input_sha256']
  for m in c['measurements'].values():
   reply=ROOT/m['reply'];assert sha(reply)==m['reply_sha256']
   raw=json.loads(subprocess.check_output([str(helper),'decode',str(reply),'measurement'],text=True))
   assert all(m[k]==v for k,v in raw.items()) and raw['digest']==c['native']['digest']
   assert raw['total_instructions']==sum(raw[k]for k in ['quantize_instructions','input_prepare_instructions','project_instructions'])
  before=c['measurements']['s1_pair_bounds']['total_instructions'];after=c['measurements']['winograd7']['total_instructions']
  assert abs(c['reduction_percent']-100*(1-after/before))<1e-10
  cases.append(dict(label=c['label'],tokens=c['tokens'],before=before,after=after,reduction_percent=c['reduction_percent']))
 files=list(dict.fromkeys([Path(__file__),ROOT/'scripts/build_s1_k2_mlp160_probe.py',ROOT/'scripts/check_s1_k2_mlp160_gate_probe.py',D/'frozen-builder.py',D/'kernel160-generator.py',D/'entry-hashes.json',D/'source-audit.json',D/'build/report.json',p]+[ROOT/p for p in b['source_hashes']]+list((D/'check-gate').glob('*.hex'))))
 summary=dict(module=r['wasm_sha256'],native_bits_equal=True,ordinary_queries=32,cases=cases,adopted=False,scope='MLP gate shape and genuine layer3 weights; arithmetic operand inputs, component only.',workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files})
 (D/'summary-gate.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-gate-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary-gate.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
