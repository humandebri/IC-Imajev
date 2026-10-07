#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-streaming-shared-probe-v1/frozen-checker.py';d=ROOT/'artifacts/s3-selective-streaming-paired-probe-v1'
 source=original.read_text().replace('s3-streaming-shared-probe-v1','s3-selective-streaming-paired-probe-v1')
 before="[(3,'current_guard_fold_k2'),(4,'rank343_k2')]";assert source.count(before)==1;source=source.replace(before,"[(3,'current_guard_fold_k2'),(4,'rank343_k2'),(5,'selective_streaming_k2')]")
 before='        results.append(case)';assert source.count(before)==1;source=source.replace(before,"        case['selective_reduction_percent']=100*(1-measured['selective_streaming_k2']['total_instructions']/before)\n"+before)
 source=source.replace("reduction_percent=case['reduction_percent'])),flush=True)","reduction_percent=case['reduction_percent'],selective_after=measured['selective_streaming_k2']['total_instructions'],selective_reduction_percent=case['selective_reduction_percent'])),flush=True)")
 source=source.replace('len(results)*2','len(results)*3')
 frozen=d/'frozen-checker.py';assert not frozen.exists();frozen.write_text(source)
 (d/'checker-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
