#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-streaming-shared-probe-v1/frozen-auditor.py';d=ROOT/'artifacts/s3-selective-streaming-paired-probe-v1'
 source=original.read_text().replace('s3-streaming-shared-probe-v1','s3-selective-streaming-paired-probe-v1').replace("len(b['patches'])==31","len(b['patches'])==32").replace("r['ordinary_queries']==42","r['ordinary_queries']==63").replace('all75_saved_replies_redecoded','all96_saved_replies_redecoded')
 before="reduction=100*(1-after/before);assert reduction==case['reduction_percent'];changes.append(dict(label=case['label'],before=before,after=after,reduction_percent=reduction))"
 after="reduction=100*(1-after/before);assert reduction==case['reduction_percent'];selected=measures['selective_streaming_k2']['total_instructions'];selected_reduction=100*(1-selected/before);assert selected_reduction==case['selective_reduction_percent'];changes.append(dict(label=case['label'],before=before,after=after,reduction_percent=reduction,selective_after=selected,selective_reduction_percent=selected_reduction))"
 assert source.count(before)==1;source=source.replace(before,after)
 before=" files=[Path(__file__)";assert source.count(before)==1;source=source.replace(before," assert len(replies)==96\n"+before)
 frozen=d/'frozen-auditor.py';assert not frozen.exists();frozen.write_text(source)
 (d/'auditor-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
