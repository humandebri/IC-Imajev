#!/usr/bin/env python3
"""Audit measured final-two-query costs; no inference or canister mutation.

A sum is not a fused-query measurement: codec work, extra copies and CDK costs
change at fusion. Preserve intermediate layer30 hidden and final KV/readout.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);ap.add_argument('--source',default='artifacts/prefix_codec/full-address-reuse-proof');a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True);out=[]
for label in ['617','insufficient','maximum']:
 p=ROOT/a.source/label/'report.json';raw=p.read_bytes();r=json.loads(raw);assert r['query_count']==len(r['queries'])
 q=r['queries'][-2:];assert q[0]['op']=='mlp_full_integer' and q[0]['tensor']=='model.language_model.layers.30.post_attention_layernorm.weight';assert q[1]['op']=='terminal_attention_mlp_integer' and q[1]['tensor']=='model.language_model.layers.31.self_attn.q_proj.weight'
 total=sum(x['ok']['instructions'] for x in q);out.append(dict(label=label,source=str(p.relative_to(ROOT)),source_sha256=hashlib.sha256(raw).hexdigest(),total_tokens=r['tokens'],suffix_tokens=r['processed_tokens'],module=r['wasm_sha256'],current_queries=r['query_count'],tail_queries=2,tail_measured_instructions=total,over_5B=total-5_000_000_000,tail_candid_bytes=sum(x['ok']['request_bytes']+x['ok']['reply_bytes'] for x in q),target_if_verified=62,scope='Counter sum only; no fused query implemented or measured. Must include decode/encode, typed decision, retained layer30 hidden and layer31 KV; do not discard input tokens.'))
(d/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
