#!/usr/bin/env python3
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];full=json.loads((ROOT/'artifacts/reference-orders.json').read_text())['records'];served=json.loads((ROOT/'artifacts/reference-serving.json').read_text())['records'];cases=json.loads((ROOT/'benchmarks/cases.json').read_text());bench={c['id']:c for c in cases}
rows=[]
for r in served:
 same=next(x for x in full if (x['id'],x['offset'])==(r['id'],r['offset']));rows.append({'id':r['id'],'offset':r['offset'],'gold':r['gold'],'value':r['result']['value'],'scores':r['result']['scores'],'tokens':r['metadata']['input_tokens'],'raw_candidate_logits':r['metadata']['logits'],'hidden_sha256':hashlib.sha256(json.dumps(r['hidden']).encode()).hexdigest(),'path_probability_max_difference':max(abs(r['result']['scores'][k]-same['result']['scores'][k])for k in r['result']['scores']),'path_label_agrees':r['result']['value']==same['result']['value']})
summary={'scope':'23 diagnostic forwards; cyclic option permutations are separate rotations=1 requests; unknown remains last. No general accuracy claim.','official_serving_commit':'a0134749e0900189c129cd6bb5000969f3b64bb5','model_lock_sha256':hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),'reference_files':{f:hashlib.sha256((ROOT/'artifacts'/f).read_bytes()).hexdigest()for f in ['reference-first.json','reference-orders.json','reference-serving.json']},'benchmark_cases_sha256':hashlib.sha256((ROOT/'benchmarks/cases.json').read_bytes()).hexdigest(),'records':rows,'controlled_unpermuted_correct':sum(('__unknown__'if r['result']['value']is None else r['result']['value'])==r['gold']for r in served if r['gold']is not None and r['offset']==0),'controlled_unpermuted_total':sum(r['gold']is not None and r['offset']==0 for r in served)}
(ROOT/'docs/host-results.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ['canister-check','projection-check']:
 p=ROOT/f'artifacts/{name}/report.json';r=json.loads(p.read_text());r['raw_report_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['raw_report_path']=str(p.relative_to(ROOT));r['max_query_instructions']=max(q['ok']['instructions']for q in r['queries']);r['max_observed_heap_bytes']=max(q['ok']['heap_pages']*65536 for q in r['queries']);r['query_count']=len(r['queries']);(ROOT/f'docs/{name}.json').write_text(json.dumps(r,indent=2)+'\n')

contract=ROOT/'artifacts/contracts/report.json'
c=json.loads(contract.read_text());c['raw_report_sha256']=hashlib.sha256(contract.read_bytes()).hexdigest();(ROOT/'docs/contracts.json').write_text(json.dumps(c,indent=2)+'\n')
files=list((ROOT/'crates').rglob('*.rs'))+list((ROOT/'canisters').rglob('*.rs'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'client').glob('*.py'))+[ROOT/'Cargo.lock',ROOT/'requirements.lock']
(ROOT/'docs/implementation-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(files)},indent=2)+'\n')

upload=ROOT/'artifacts/upload-check/report.json'
u=json.loads(upload.read_text());u['raw_report_sha256']=hashlib.sha256(upload.read_bytes()).hexdigest();(ROOT/'docs/upload-check.json').write_text(json.dumps(u,indent=2)+'\n')
