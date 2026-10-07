#!/usr/bin/env python3
"""Recheck reusable-weight diagnostic outputs and saved inclusive profiles."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_update_other_profile.py'
 source=p.read_text().replace('artifacts/update-other-profile-v1','artifacts/update-other-profile-v3')
 exec(compile(source,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/update-other-profile-v3'
 proof=json.loads((d/'proof/report.json').read_text())
 assert proof['banks'][1]['preparation']['update_calls_this_run']==0
 assert proof['banks'][1]['preparation']['initial']==proof['banks'][1]['preparation']['final']
 checked=[]
 for bank in proof['banks']:
  for c in bank['measurement']['cases']:
   directory=d/'proof'/bank['bank']/'measurement'/f"{c['case']}-r{c['repeat']}"
   row=json.loads(sorted(directory.glob('*.json'))[0].read_text())
   raw=directory/'active-reset.candid'
   assert row['active_reset_rejected'] and hashlib.sha256(raw.read_bytes()).hexdigest()==row['active_reset_reply_sha256']
   text=raw.read_text();assert 'Err' in text and 'inference is active' in text
   checked.append(c['case'])
 assert set(checked)=={'617','620','653'}
 summary=json.loads((d/'profile-summary.json').read_text())
 summary.update(active_reset_rejections_verified=checked,common_weight_preparation_updates=0,fixed_weights_reused=True)
 (d/'profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
