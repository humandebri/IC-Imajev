#!/usr/bin/env python3
"""Paired query/update metering, raw nat64 replies with exact counter deltas."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1];CANISTER='4xhad-gd777-77775-aaacq-cai'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/simd-meter-calibration-v3';b=json.loads((d/'build.json').read_text());entry=json.loads((d/'builder-entry-hashes.json').read_text());check=d/'check';check.mkdir(exist_ok=False)
 for group in [b['source_hashes'],entry]:
  for p,h in group.items():assert sha(ROOT/p)==h,p
 def status():return json.loads(subprocess.check_output(['icp','canister','status',CANISTER,'--network','local','--identity','imajev-local','--json'],text=True))
 assert status()['module_hash'].removeprefix('0x')==b['module']
 methods=['empty','v128_get','v128_set','dot','shuffle','load32_splat','load128','load8x8_s','load128_twice','load128_four','quad_splats','quad_packet']
 values={};saved=[]
 for mode in ['query','update']:
  for name in methods+['quad_splats','quad_packet']:
   did=d/('probe.did'if mode=='query'else'probe-update.did')
   cmd=['icp','canister','call',CANISTER,name,'()','--network','local','--identity','imajev-local','--candid',str(did),'--output','hex']
   if mode=='query':cmd+=['--query']
   raw=subprocess.check_output(cmd,text=True);p=check/f'{len(saved):03d}-{mode}-{name}.hex';p.write_text(raw)
   data=bytes.fromhex(raw.strip().removeprefix('0x'));assert len(data)==15 and data[:7]==b'DIDL\0\1\x78';v=int.from_bytes(data[7:],'little');key=mode+'/'+name
   assert key not in values or values[key]==v;values[key]=v
   saved.append(dict(mode=mode,method=name,instructions=v,reply=str(p.relative_to(ROOT)),reply_sha256=sha(p)))
   print(json.dumps(dict(mode=mode,method=name,instructions=v)),flush=True)
 assert status()['module_hash'].removeprefix('0x')==b['module']
 def frac(v):f=Fraction(v,10000);return dict(numerator=f.numerator,denominator=f.denominator,value=float(f))
 deltas={key:frac(v-values[key.split('/')[0]+'/empty'])for key,v in values.items()}
 packets={mode:frac(values[mode+'/quad_packet']-values[mode+'/quad_splats'])for mode in ['query','update']}
 files=[Path(__file__),d/'build.json',d/'builder-entry-hashes.json']+[ROOT/p for p in b['source_hashes']]+[ROOT/p for p in entry]+[ROOT/s['reply']for s in saved]
 result=dict(complete=True,module=b['module'],measurements=values,loop_deltas=deltas,packet_extra_per_group=packets,
  query_update_all_equal=all(values['query/'+n]==values['update/'+n]for n in methods),replies=saved,
  source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},scope='Matched loop bodies only, dual IC method exports on one module. No model inference or full goal claim.')
 (d/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(packet_extra_per_group=packets,query_update_all_equal=result['query_update_all_equal'])))
if __name__=='__main__':main()
