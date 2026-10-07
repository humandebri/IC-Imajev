#!/usr/bin/env python3
"""Check instrumented S1 against native integers and save four disjoint counters."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/check_s1_output160.py';source=p.read_text()
 source=source.replace('artifacts/s1-output160-v1/build','artifacts/s1-inner-profile-v1/build-v6').replace('artifacts/s1-output160-v1/check','artifacts/s1-inner-profile-v1/check')
 source=source.replace('output160','instrumented_s1').replace('streamed rank49/output64','instrumented exact S1')
 anchor="        after = measured['instrumented_s1']['total_instructions']"
 # The upstream uses both assignments on one line; insert after measurement.
 anchor="        before = measured['s1_pair_bounds']['total_instructions']; after = measured['instrumented_s1']['total_instructions']"
 assert source.count(anchor)==1
 insertion="""        profile_reply = ROOT/measured['instrumented_s1']['reply']
        inner = json.loads(subprocess.check_output([str(B/'tool'),'decode',str(profile_reply)],text=True))
        assert len(inner)==4 and all(v>=0 for v in inner)
        assert inner[0]>0 and inner[3]>0
        assert (inner[1]>0)==(n>=4)
        assert (inner[2]>0)==(n>=3 and n%2==1)
        assert sum(inner)<=measured['instrumented_s1']['project_instructions']
        measured['instrumented_s1']['inner_profile']=inner
"""
 source=source.replace(anchor,insertion+anchor)
 d=ROOT/'artifacts/s1-inner-profile-v1';(d/'frozen-check.py').write_text(source)
 (d/'check-upstream.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')
 exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
