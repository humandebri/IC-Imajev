#!/usr/bin/env python3
"""Read-only comparison: structural MACs and actual reports; no inference in Laya."""
import collections,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];LAYA=pathlib.Path('/Volumes/KINGSTON/ICP/IC-Laya-Standalone')
i=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());l=json.loads((LAYA/'checkpoints/laya-int8/manifest.json').read_text());r=json.loads((ROOT/'docs/fused-results.json').read_text());ops=collections.Counter()
for q in r['query_metrics']:ops[q['op']]+=q['instructions']
# Exclude embedding lookup, depthwise convolution, small gate projections and norm vectors.
iweights=[t for t in i['tensors'] if '.layers.' in t['name'] and t['name'].endswith('.weight') and '.lora_' not in t['name'] and t['cols']>4 and t['rows']>1]
lweights=[t for t in l['tensors'] if t['name'].startswith('encoder.') and t['name'].endswith('.weight') and len(t['shape'])==2]
i_per=sum(t['rows']*t['cols'] for t in iweights);l_per=sum(math.prod(t['shape']) for t in lweights)
# Exact multiplication count for the executed LoRA projection tile plan, including repeated A.
planned_base=planned_a=planned_b=0;unique_a={}
for q in r['query_metrics']:
 if q['op']!='lora_project':continue
 b=(ROOT/f"artifacts/activation-int8-fused-617/queries/{q['index']:06d}.request.bin").read_bytes();h=json.loads(b[4:4+int.from_bytes(b[:4],'little')]);n,rows,cols=h['dims'][:3];rank=64
 planned_base+=n*rows*cols;planned_a+=n*rank*cols;planned_b+=n*rows*rank
 # Each projection/token chunk has identical input hash; row tiles repeat the same A.
 header_len=int.from_bytes(b[:4],'little');payload_hash=hashlib.sha256(b[4+header_len:-32]).hexdigest();unique_a[(h['tensor'],n,cols,payload_hash)]=n*rank*cols
refs=['docs/INT8_QUERY_OPTIMIZATION_SEARCH.md','docs/INT8_OPTIMIZATION_V4.md','crates/laya-candle/src/int8.rs','checkpoints/laya-int8/manifest.json']
summary=dict(imajev_tokens=132,laya_report_tokens=128,imajev_handler_instructions=r['total_instructions'],laya_report_int8_handler_instructions=42129893539,laya_report_queries=17,imajev_queries=r['query_count'],observed_instruction_ratio=r['total_instructions']/42129893539,comparison_caveat='Different model, input, graph, pack and instrumentation; structural explanation, not same-question accuracy or speed benchmark',imajev_dense_base_macs_per_token=i_per,laya_encoder_dense_macs_per_token=l_per,structural_ratio=i_per/l_per,imajev_macs_at132=i_per*132,laya_encoder_macs_at128=l_per*128,operation_instructions=dict(ops),projection_fraction=ops['lora_project']/r['total_instructions'],lora_plan=dict(base_macs=planned_base,a_macs=planned_a,b_macs=planned_b,a_macs_if_reused=sum(unique_a.values()),redundant_a_macs=planned_a-sum(unique_a.values()),redundant_a_fraction_of_projection_macs=(planned_a-sum(unique_a.values()))/(planned_base+planned_a+planned_b)),laya_readonly_reference_hashes={p:hashlib.sha256((LAYA/p).read_bytes()).hexdigest() for p in refs})
(ROOT/'docs/laya-cost-analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k not in ('laya_readonly_reference_hashes','operation_instructions')})
