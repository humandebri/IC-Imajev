#!/usr/bin/env python3
"""Re-decode all saved Candid chunks and repeat the independent dense recurrence."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
from check_delta_capture import check
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/delta-capture-v2';P=D/'proof'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((P/'report.json').read_text());assert r['complete'] and r['baseline_restored'] and len(r['captures'])==72 and len(r['cases'])==3
 build=json.loads((D/'build/report.json').read_text())
 normal=ROOT/'artifacts/update-quantize-cached-v1/build'
 normal_report=json.loads((normal/'report.json').read_text())
 assert len(build['patches'])==len(normal_report['patches'])==10
 assert all(a['source_sha256']==b['source_sha256']for a,b in zip(build['patches'],normal_report['patches']))
 for p in (normal/'runtime').glob('*.rs'):
  text=(D/'build/runtime'/p.name).read_text()
  if p.name=='lib.rs':
   suffix='\npub mod delta_capture;\n';assert text.endswith(suffix);text=text[:-len(suffix)]
  elif p.name=='delta_full_log.rs':
   for line in [' crate::delta_capture::begin(&states,n,key_major,r.step);\n','  crate::delta_capture::head(head,&qh,&kh,&vh,&gh,&bh,state,&values);\n']:
    assert text.count(line)==1;text=text.replace(line,'')
  assert text==p.read_text(),('unexpected runtime change',p.name)
 for hashes in [r['source_hashes'],build['source_hashes'],build['dependency_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 restored=json.loads((P/'restored.json').read_text());assert restored['cache_equal'] and restored['pack_equal'] and restored['snapshot_deleted']
 evidence={};verified=[];temporary=P/'redecoded.bin'
 try:
  for row in r['captures']:
   capture=Path(row['path']);assert sha(capture)==row['sha256'];assert all(sha(ROOT/p)==h for p,h in row['reply_hashes'].items())
   pieces=[]
   for path in sorted(capture.parent.glob('chunk-*.reply.hex')):
    subprocess.run([str(P/'decode'),str(path),str(temporary)],check=True);b=temporary.read_bytes()
    offset=int(path.name.split('-')[1].split('.')[0]);assert offset==sum(map(len,pieces));pieces.append(b)
    evidence[str(path.relative_to(ROOT))]=sha(path)
   assert b''.join(pieces)==capture.read_bytes()
   subprocess.run([str(P/'decode'),str(capture.parent/'header.reply.hex'),str(temporary)],check=True)
   assert temporary.read_bytes()==capture.read_bytes()[:24]
   header=capture.parent/'header.reply.hex';evidence[str(header.relative_to(ROOT))]=sha(header)
   actual=check(capture)
   for k,v in actual.items():assert row[k]==v,(row['case'],k)
   verified.append(dict(case=row['case'],layer=row['layer'],dense_values=row['dense_values'],output_values=row['output_values']))
   evidence[str(capture.relative_to(ROOT))]=sha(capture)
 finally:temporary.unlink(missing_ok=True)
 for name in ['617','620','653']:assert sorted(v['layer']for v in verified if v['case']==name)==[i for i in range(32)if i%4!=3]
 note=D/'progress-note.md'
 if not note.exists():note.write_text('Delta capture verification: see summary.json for the module, compared cases, evidence hashes and comparison limits. This capture does not establish a performance improvement.\n')
 files=[Path(__file__),ROOT/'scripts/check_delta_capture.py',ROOT/'scripts/prove_delta_capture.py',ROOT/'scripts/delta_capture_args.rs',ROOT/'scripts/delta_capture.rs',ROOT/'scripts/build_delta_capture.py',D/'frozen-builder.py',D/'workflow-hashes.json',D/'build/report.json',P/'report.json',P/'sources.json',P/'restored.json',note]
 summary=dict(complete=True,baseline_restored=True,module=r['module'],cases=verified,dense_state_values_checked=sum(v['dense_values']for v in verified),saved_candid_chunks_redecoded=True,dense_state_bits_equal=True,pre_bf_output_bits_equal=True,runtime_changes_only_capture_hooks=True,ten_kernel_sources_equal=True,evidence_hashes=evidence,workflow_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},missing_historical_hidden_layers=[30],performance_claim=False,goal_complete=False)
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[D/'summary.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps({k:v for k,v in summary.items()if k not in ['cases','evidence_hashes','workflow_hashes']},indent=2))
if __name__=='__main__':main()
