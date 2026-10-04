#!/usr/bin/env python3
"""Archive checked wide-output evidence and compare to the prior codec candidate."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];d=ROOT/'artifacts/prefix_codec/full-s1-wide-proof';r=json.loads((d/'report.json').read_bytes());b=json.loads((ROOT/'artifacts/prefix_codec/full-host-checksum-proof/report.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/p)==v for p,v in r['source_hashes'].items())
changes=[]
for c in r['cases']:
 old=next(x for x in b['cases']if x['label']==c['label']);assert c['full_bitwise_equal'];assert c['query_count']==old['query_count']-(1 if c['label']=='maximum' else 0);assert c['total_candid_bytes']==old['total_candid_bytes']-(1373665 if c['label']=='maximum' else 0)
 saved=old['total_instructions']-c['total_instructions'];assert saved>0
 changes.append(dict(label=c['label'],query_count=c['query_count'],before_instructions=old['total_instructions'],after_instructions=c['total_instructions'],saved_instructions=saved,saved_percent=100*saved/old['total_instructions'],candid_bytes=c['total_candid_bytes'],max_query_instructions=c['max_query_instructions'],wall_seconds=c['wall_seconds'],heap_bytes=c['max_observed_heap_bytes']))
with zipfile.ZipFile(d/'validated-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for p in r['source_hashes']:z.write(ROOT/p,p)
 for p in ['scripts/generate_wat_s1_wide.py','scripts/build_s1_wide.py','scripts/check_s1_wide.py','scripts/check_terminal_tail.py','scripts/test_terminal_tail_client.py','scripts/archive_s1_output_proof.py']:z.write(ROOT/p,p)
(d/'before-after.json').write_text(json.dumps(dict(cases=changes,baseline_report_sha256=sha(ROOT/'artifacts/prefix_codec/full-host-checksum-proof/report.json'),candidate_report_sha256=sha(d/'report.json')),indent=2)+'\n');print(json.dumps(changes,indent=2))
