#!/usr/bin/env python3
"""Audit actual paid full-Delta row sharing, saved Candid and all32 hidden."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/report_paid_finite_max.py';s=p.read_text().replace('artifacts/paid-finite-max-v1','artifacts/paid-phase-profile-v2').replace('artifacts/update-finite-max-v1','artifacts/update-phase-profile-v2')
 old="dict(changed_runtime_files=['finite_simd.rs'],predicate_sites_equal=True,arithmetic_kernels_equal=True)"
 new="{'changed_runtime_files': ['delta_full_log.rs', 'lib.rs', 'profile.rs'], 'arithmetic_kernels_equal': True, 'scope': 'Diagnostic only. Four coarse actual compute phase spans: every matrix_loaded call, base INT8 projection, activation quantization and outer Delta recurrence. Deep kernel/loop profiling disabled. One outer Delta recurrence span added. Every original22 WAT retained. Profiled total is not used as an optimization result.'}"
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('Dense Delta arithmetic kernels unchanged from previous independently verified capture module.','Dense Delta kernels unchanged; diagnostic phase counters validated against saved hidden/state references.')
 s=s.replace('extras=[Path(__file__),p,',"extras=[Path(__file__),p,ROOT/'scripts/report_paid_finite_max.py',d/'frozen-row-reporter.py',")
 (ROOT/'artifacts/paid-phase-profile-v2/frozen-row-reporter.py').write_text(s)
 exec(compile(s,str(p),'exec'),dict(__file__=__file__,__name__='__main__'))
 d=ROOT/'artifacts/paid-phase-profile-v2';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tool=ROOT/'artifacts/phase-profile-tools-v1/decode';tb=json.loads((tool.parent/'report.json').read_text());assert h(tool)==tb['binary_sha256'];assert all(h(ROOT/p)==v for p,v in {**tb['source_hashes'],**tb['dependency_hashes']}.items())
 proof=json.loads((d/'proof/report.json').read_text());summary=json.loads((d/'summary.json').read_text());allowed={'base_project_inclusive','f32_project_inclusive','activation_quantize','delta_recurrence'};cases=[];extras=[tool,tool.parent/'report.json',tool.parent/'phase-profile.did',tool.parent/'manual-text-fixture.hex',tool.parent/'manual-text-fixture.json',d/'scheduler-source-audit.json']
 for result in proof['results']:
  trace=result['phase_profile'];reply=ROOT/trace['reply'];assert h(reply)==trace['reply_sha256'];rows=json.loads(subprocess.check_output([str(tool),str(reply)],text=True));assert rows==trace['rows'];assert rows and all(k in allowed and n>0 and i>=0 for k,i,n in rows);aggregated={}
  for label,instructions,count in rows:
   v=aggregated.setdefault(label,dict(instructions=0,calls=0));v['instructions']+=instructions;v['calls']+=count
  total=sum(v['instructions']for v in result['row']['result']['Ok']['workers']);measured=sum(v['instructions']for v in aggregated.values());assert measured==trace['instructions'] and measured<=total
  cases.append(dict(case=result['case'],phases=aggregated,worker_instructions=total,phase_instructions=measured,unattributed_instructions=total-measured));extras.append(reply)
 summary.update(diagnostic_only=True,adopted=False,profile_totals_are_not_optimization_results=True,phase_candid_redecoded=True,phase_cases=cases);summary['workflow_hashes'].update({str(p.relative_to(ROOT)):h(p)for p in extras});(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 archive=d/'frozen-paid-proof.zip';temporary=d/'frozen-paid-proof-profile.zip'
 with zipfile.ZipFile(archive)as old,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED)as new:
  latest={v.filename:v for v in old.infolist()};names=set(latest);summary_name=str((d/'summary.json').relative_to(ROOT))
  for name,info in latest.items():
   if name!=summary_name:new.writestr(name,old.read(info))
  for p in extras:
   name=str(p.relative_to(ROOT))
   if name not in names:new.write(p,name);names.add(name)
  new.write(d/'summary.json',summary_name)
 temporary.replace(archive)
 with zipfile.ZipFile(archive)as z:assert len(z.namelist())==len(set(z.namelist())) and z.read(summary_name)==(d/'summary.json').read_bytes()
 print(json.dumps(dict(diagnostic_only=True,phase_cases=cases),indent=2))

if __name__=='__main__':main()
