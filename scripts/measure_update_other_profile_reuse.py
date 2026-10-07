#!/usr/bin/env python3
"""Read stage profiles from the diagnostic with reusable immutable weights."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/measure_update_instruction_profile.py'
 source=p.read_text().replace('artifacts/update-instruction-profile-v1','artifacts/update-other-profile-v3')
 anchor='    source = source.replace(anchor, replacement)'
 assert source.count(anchor)==1
 addition='''    replacement += '''+repr('''
    if len(rows)==0 and not progress['done']:
     reset_did=d/'reset.did';reset_did.write_text('service : {reset_update_prefix:()->(variant {Ok;Err:text});}')
     raw_reset=subprocess.check_output(['icp','canister','call',report['canister'],'reset_update_prefix','()','--network','local','--identity','imajev-local','--candid',str(reset_did),'--output','candid'],text=True,cwd=ROOT)
     assert 'Err' in raw_reset and 'inference is active' in raw_reset
     reset_path=directory/'active-reset.candid';reset_path.write_text(raw_reset)
     r['active_reset_reply_sha256']=sha(reset_path)
     r['active_reset_rejected']=True
    ''')+'\n'
 source=source.replace(anchor,addition+anchor)
 exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
