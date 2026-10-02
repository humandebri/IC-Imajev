#!/usr/bin/env python3
"""Count duplicate LoRA A work for identical wire inputs; no predicted instruction saving claim."""
import argparse,collections,hashlib,json,pathlib,struct
ROOT=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--directory',default='artifacts/prefix-hit-617');a=p.parse_args();r=json.loads((ROOT/a.directory/'first-report.json').read_text());m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());weights={t['name']:t for t in m['tensors']};groups=collections.Counter()
for q in r['queries']:
 if q['op'] not in ['lora_integer','mlp_gate_up_integer']:continue
 b=(ROOT/a.directory/f"queries/{q['index']:06d}.request.bin").read_bytes();n,=struct.unpack('<I',b[:4]);h=json.loads(b[4:4+n]);digest=hashlib.sha256(b[4+n:-32]).hexdigest()
 names=h['aux'][:1] if q['op']=='lora_integer' else [h['tensor'].replace('.weight','.lora_A.weight'),h['aux'][0].replace('.weight','.lora_A.weight')]
 for name in names:groups[name,h['dims'][0],h['dims'][2],digest]+=1
saved_mac=sum((count-1)*n*cols*weights[name]['rows'] for (name,n,cols,digest),count in groups.items());saved_reads=sum((count-1)*cols*weights[name]['rows']*4 for (name,n,cols,digest),count in groups.items());axis_bytes=sum(n*weights[name]['rows']*4 for (name,n,cols,digest),count in groups.items() if count>1)
report=dict(scope='Actual identical input groups; theoretical avoidable MAC/read bytes only, not measured instructions or implemented reuse',source=a.directory,repeated_input_groups=sum(v>1 for v in groups.values()),duplicate_A_evaluations=sum(v-1 for v in groups.values()),avoidable_A_MACs=saved_mac,avoidable_A_weight_read_bytes=saved_reads,minimum_F32_axis_bytes_if_each_repeated_group_is_exported_once=axis_bytes,reuse_implemented=False)
(ROOT/'docs/lora-reuse-candidate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
