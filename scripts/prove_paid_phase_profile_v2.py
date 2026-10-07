#!/usr/bin/env python3
"""Collect actual paid phase traces while verifying full saved references and paid API."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def patch_profile_proof(s):
 anchor='    results.append(dict(case=name,repeat=repeat,request=request,quote=quotes[i],row=row,debug=debug));'
 assert s.count(anchor)==1
 capture='''    phase_reply=D/f'phase-{name}-{repeat}.hex'
    phase_raw=subprocess.check_output(['icp','canister','call',TARGET,'paid_graph_profile','()','--query','--network','local','--identity','imajev-local','--candid',str(ROOT/'artifacts/phase-profile-tools-v1/phase-profile.did'),'--output','hex'],text=True,cwd=ROOT);phase_reply.write_text(phase_raw)
    phase_rows=json.loads(subprocess.check_output([str(ROOT/'artifacts/phase-profile-tools-v1/decode'),str(phase_reply)],text=True,cwd=ROOT))
    allowed={'base_project_inclusive','f32_project_inclusive','activation_quantize','delta_recurrence'}
    assert phase_rows and all(label in allowed and count>0 and instructions>=0 for label,instructions,count in phase_rows)
    phase=dict(rows=phase_rows,reply=str(phase_reply.relative_to(ROOT)),reply_sha256=sha(phase_reply),instructions=sum(v[1]for v in phase_rows))
    assert phase['instructions']<=sum(v['instructions']for v in result['workers'])
'''
 s=s.replace(anchor,capture+anchor.replace('debug=debug)','debug=debug,phase_profile=phase)'))
 anchor='paths=[Path(__file__),';assert s.count(anchor)==1;s=s.replace(anchor,"paths=[ROOT/'artifacts/phase-profile-tools-v1/report.json',ROOT/'artifacts/phase-profile-tools-v1/decode',ROOT/'artifacts/phase-profile-tools-v1/phase-profile.did',Path(__file__),")
 return s
def main():
 p=ROOT/'scripts/prove_paid_quantize_cached_v2.py';s=p.read_text().replace('artifacts/update-quantize-cached-v1','artifacts/update-phase-profile-v2').replace('artifacts/paid-quantize-cached-v2','artifacts/paid-phase-profile-v2')
 anchor="(d/'frozen-proof.py').write_text(s)";assert s.count(anchor)==1;s=s.replace(anchor,"s=patch_profile_proof(s);"+anchor)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__',patch_profile_proof=patch_profile_proof))
if __name__=='__main__':main()
