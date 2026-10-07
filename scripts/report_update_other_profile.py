#!/usr/bin/env python3
"""Verify original output proofs and rank inclusive Rust spans without overlap sums."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=ROOT/'artifacts/update-other-profile-v1'
    subprocess.run([sys.executable,str(ROOT/'scripts/report_update_f32_candidate.py'),'--directory',str(d.relative_to(ROOT))],check=True,stdout=subprocess.DEVNULL)
    verified=json.loads((d/'summary.json').read_text())
    assert verified['verified'] and verified['baseline_restored']
    cases=[]
    proof=json.loads((d/'proof/report.json').read_text())
    for bank in proof['banks']:
        for c in bank['measurement']['cases']:
            counts,calls=Counter(),Counter()
            path=d/'proof'/bank['bank']/'measurement'/f"{c['case']}-r{c['repeat']}"
            for reply in sorted(path.glob('*.json')):
                row=json.loads(reply.read_text())
                raw=reply.with_suffix('.profile.candid')
                assert sha(raw)==row['profile_reply_sha256']
                text=raw.read_text();decoded,_=json.JSONDecoder().raw_decode(text[text.index('"'):])
                spans=json.loads(decoded);assert spans==row['instruction_profile']
                for name,value,n in spans:
                    assert value>=0 and n>0
                    counts[name]+=value;calls[name]+=n
            assert counts['base_project_inclusive']>0 and counts['delta_head_inclusive']>0
            ranking=[dict(name=name,instructions=value,calls=calls[name]) for name,value in counts.most_common()]
            cases.append(dict(case=c['case'],handler_instructions=c['total_handler_instructions'],inclusive_spans=ranking))
    assert {c['case'] for c in cases}=={'617','620','653'}
    result=dict(module=verified['module'],proof_verified=True,baseline_restored=True,cases=cases,
                scope='Diagnostic Rust call-boundary counters with profiling overhead. Spans can nest and are not summed as exclusive costs. Not a production improvement measurement.')
    (d/'profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    verified['scope']=result['scope'];verified['diagnostic_only']=True
    (d/'summary.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
