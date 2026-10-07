#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'scripts/report_s2_k2_leaf_outer_probe.py';d=ROOT/'artifacts/s2-contiguous-roots-probe-v1'
 source=original.read_text().replace('s2-k2-leaf-outer-probe-v1','s2-contiguous-roots-probe-v1').replace('s2-k2-leaf-outer-kernels-v1','s2-contiguous-roots-kernels-v1').replace('layout-audit.json','execution-report.json').replace('rank49_leaf_outer','rank49_contiguous_roots').replace('entry-builder-hashes.json','builder-entry-hashes.json').replace('entry-checker-hashes.json','checker-entry-hashes.json').replace('auditor-entry-hashes.json','execution-entry-hashes.json').replace("d/'frozen-workflow.zip','w'","d/'frozen-workflow.zip','x'")
 frozen=d/'frozen-reporter.py';assert not frozen.exists();frozen.write_text(source)
 (d/'reporter-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
