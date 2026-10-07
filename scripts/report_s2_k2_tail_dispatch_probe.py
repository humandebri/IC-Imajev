#!/usr/bin/env python3
"""Independent saved-reply audit of rank49 K2 versus actual current full K2 control."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/s2-k2-tail-dispatch-probe-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'scripts/report_s2_pair_late_probe.py';s=p.read_text().replace('s2-pair-late-v1','s2-k2-tail-dispatch-probe-v1').replace('check-v2/report.json','check/report.json').replace('s2_pair_late','rank49_k2').replace("len(r['cases']) == 19","len(r['cases']) == 21").replace('ordinary_queries=38','ordinary_queries=42')
 (D/'frozen-reporter.py').write_text(s);exec(compile(s,str(D/'frozen-reporter.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 summary=json.loads((D/'summary.json').read_text());b=json.loads((D/'build/report.json').read_text());check=json.loads((D/'check/report.json').read_text())
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in b[key].items())
 assert summary['module']==b['wasm_sha256']==sha(D/'build/diagnostic.wasm')and check['ordinary_queries']==42
 k=ROOT/'artifacts/s2-k2-tail-dispatch-kernels-v1';audit=json.loads((k/'layout-audit.json').read_text());assert audit['all_integer_roots_equal']and audit['actual_emitted_transpose_masks_verified']
 for p,h in audit['source_hashes'].items():assert sha(ROOT/p)==h
 for case in check['cases']:assert sha(D/'check'/f"{case['label']}.input.bin")==case['input_sha256']
 files=list(dict.fromkeys([Path(__file__),ROOT/'scripts/build_s2_k2_tail_dispatch_probe.py',ROOT/'scripts/check_s2_k2_tail_dispatch_probe.py',D/'frozen-reporter.py',D/'build/report.json',D/'check/report.json',D/'source-audit.json',k/'layout-audit.json']+[ROOT/p for p in b['source_hashes']]+list((D/'check').glob('*.hex'))))
 summary.update(current_full_k2_runtime_control=True,rank=49,output_tiles=[80,16],integer_layout_identity_verified=True,emitted_zero_leaf_thresholds_verified=audit['emitted_zero_leaf_thresholds_verified'],pruned_leaves_per_tail=audit['pruned_leaves_per_tail'],single_token_current_k2_fallback=True,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Actual layer3 Q weights and saved/synthetic operands. Component only, includes input preparation. Multiple token rank49; single token current K2 fallback. No full inference claim.')
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(module=summary['module'],cases=len(summary['cases']),native_bits_equal=True)))
if __name__=='__main__':main()
