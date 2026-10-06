#!/usr/bin/env python3
"""Verify archived S2 wide query evidence and summarize the rejected candidate."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/winograd2-register-v2';r=json.loads((d/'check/report.json').read_text());helper=ROOT/'artifacts/bounded_i16/native/release/wat_s1_args'
 assert all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 assert len(r['cases'])==19
 for c in r['cases']:
  assert c['bitwise_equal']
  for key in ['s1_pair_bounds','s2_wide_stream']:
   m=c['measurements'][key];path=ROOT/m['reply'];assert sha(path)==m['reply_sha256']
   raw=json.loads(subprocess.check_output([str(helper),'decode',str(path),'measurement'],text=True))
   for field,value in raw.items():assert m[field]==value
 report=dict(check_report_sha256=sha(d/'check/report.json'),wasm_sha256=r['wasm_sha256'],ordinary_queries=38,all_kernel_bits_equal=True,adopted=False,reason='Transformation/reconstruction and streaming overhead exceeded dot savings.',cases=[dict(label=c['label'],tokens=c['tokens'],reduction_percent=c['reduction_percent']) for c in r['cases']])
 (d/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':main()
