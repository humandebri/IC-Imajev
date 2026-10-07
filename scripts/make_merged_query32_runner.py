#!/usr/bin/env python3
"""Freeze a private copy of the query32 runner for this paired experiment."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 src=ROOT/'scripts/run_prefix_canister.py';s=src.read_text()
 build=json.loads((ROOT/'artifacts/merged-query32-v1/build/report.json').read_text());module=build['wasm_sha256']
 s=s.replace("ap.add_argument('--query32-start'","ap.add_argument('--step-offset',type=int,default=0);ap.add_argument('--query32-start'",1)
 s=s.replace("   'f6b399",f"   '{module}',\n   'f6b399",1)
 needle="  cache_module='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'"
 s=s.replace(needle,needle+f"\n  if m['pack_hash']=='{build['merged_pack_hash']}':cache_module='{module}'",1)
 begin=s.index(' if args.query32_start:\n  from template_prefix import select');end=s.index(' cache_start=',begin)
 s=s[:begin]+" if args.query32_start:prefix_bank='explicit-experiment-cache'\n"+s[end:]
 s=s.replace("wasm_hash=='2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05'",f"wasm_hash in ('2cfe5d3a1d2da9481d9446dbd0a877617de8db140f8347058b02a1dbb2c63a05','{module}')")
 s=s.replace(" path=directory/'session.json'\n",' if not 0<=args.step_offset<2**63:raise ValueError("step offset bounds")\n session["request_step_offset"]=args.step_offset\n'+" path=directory/'session.json'\n",1)
 s=s.replace(' if args.reuse_projection_inputs:\n',' t.index=args.step_offset\n if args.reuse_projection_inputs:\n',1)
 s=s.replace('    from query32_templates import Query32PrefixGraph','    import query32_templates\n    query32_templates.MODULE=wasm_hash\n    from query32_templates import Query32PrefixGraph',1)
 p=ROOT/'scripts/run_merged_query32.py'
 if p.exists():assert p.read_text()==s,'Existing experimental runner differs from generated copy'
 else:p.write_text(s)
 (ROOT/'artifacts/merged-query32-v1/runner-provenance.json').write_text(json.dumps(dict(source=str(src.relative_to(ROOT)),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),module=module,scope=__doc__),indent=2)+'\n')
if __name__=='__main__':main()
