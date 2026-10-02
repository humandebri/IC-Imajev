#!/usr/bin/env python3
"""Weight measured identical integer shapes by actual full graph frequency, not benchmark publicity."""
import collections,json,pathlib,struct,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'artifacts/mlp-wide-full-617/first-report.json').read_text());counts=collections.Counter()
for q in r['queries']:
 if 'integer' not in q['op']:continue
 with (ROOT/f"artifacts/mlp-wide-full-617/queries/{q['index']:06d}.request.bin").open('rb') as f:
  n,=struct.unpack('<I',f.read(4));h=json.loads(f.read(n))
 counts[q['op'],tuple(h['dims'][:3])]+=1
rows=[]
for name in ['baseline','r32','r16','c8','direct','combined']:
 p=ROOT/f'artifacts/bottleneck/directions-{name}/report.json'
 if not p.exists():continue
 a=json.loads(p.read_text());total=sum(c['normal']['instructions']*counts[c['op'],tuple(c['dims'][:3])] for c in a['cases']);assert len(a['cases'])==len(counts) and all(c['bitwise_equal'] for c in a['cases'])
 rows.append(dict(candidate=name,shape_cases=len(a['cases']),weighted_projection_instructions=total,bitwise_equal=True,wasm_sha256=a['wasm_sha256'],source_sha256=hashlib.sha256((ROOT/f"artifacts/directions/{'baseline' if name=='baseline' else name}.rs").read_bytes()).hexdigest(),relative_to_baseline=total/rows[0]['weighted_projection_instructions'] if rows else 1))
report=dict(scope='Same real inputs, all distinct integer shapes; weighted shape estimate, not a full inference total',query_shapes=sum(counts.values()),candidates=rows,rejected=[dict(candidate='r8',reason='IC0522: exceeds 5,000,000,000 instructions on 132-token 5704-row fused MLP; successful smaller shapes bitwise equal; no complete weighted result')])
(ROOT/'docs/integer-layout-candidates.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
