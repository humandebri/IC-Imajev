#!/usr/bin/env python3
"""Save raw matched-loop replies and derive local runtime instruction deltas."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--canister',required=True);args=parser.parse_args()
 assert args.canister=='4xhad-gd777-77775-aaacq-cai'
 d=ROOT/'artifacts/simd-meter-calibration-v2';b=json.loads((d/'build.json').read_text());check=d/'check';check.mkdir(exist_ok=False)
 entry=json.loads((d/'builder-entry-hashes.json').read_text())
 for group in [b['source_hashes'],entry]:
  for p,h in group.items():assert sha(ROOT/p)==h,p
 def status():return json.loads(subprocess.check_output(['icp','canister','status',args.canister,'--network','local','--identity','imajev-local','--json'],text=True))
 assert status()['module_hash'].removeprefix('0x')==b['module']
 saved=[];measured={}
 for index,name in enumerate(list(b['operations'])+['empty','dot','call_v128_8192']):
  raw=subprocess.check_output(['icp','canister','call',args.canister,name,'()','--network','local','--identity','imajev-local','--candid',str(d/'probe.did'),'--output','hex','--query'],text=True)
  p=check/f'{index:03d}-{name}.hex';p.write_text(raw);data=bytes.fromhex(raw.strip().removeprefix('0x'))
  assert len(data)==15 and data[:7]==b'DIDL\0\1\x78'
  value=int.from_bytes(data[7:],'little')
  assert name not in measured or measured[name]==value
  measured[name]=value;saved.append(dict(method=name,instructions=value,reply=str(p.relative_to(ROOT)),reply_sha256=sha(p)))
  print(json.dumps(dict(method=name,instructions=value)),flush=True)
 assert status()['module_hash'].removeprefix('0x')==b['module']
 def delta(name,baseline='empty'):
  n=b['method_iterations'][name];assert n==b['method_iterations'][baseline]
  f=Fraction(measured[name]-measured[baseline],n);return dict(numerator=f.numerator,denominator=f.denominator,value=float(f))
 deltas={name:delta(name,'empty100'if b['method_iterations'][name]==100 else 'empty')for name in b['operations']}
 locals_costs={name:delta(name,'call_empty')for name in b['operations']if name.startswith('call_')}
 # Compound bodies include their operand reads and result write; these deltas
 # are direct measurements, not absolute per-op costs inferred from drop=0.
 files=[Path(__file__),d/'build.json',d/'builder-entry-hashes.json']+[ROOT/p for p in b['source_hashes']]+[ROOT/p for p in entry]+[ROOT/s['reply']for s in saved]
 report=dict(complete=True,module=b['module'],all_raw_nat64_replies_redecoded=True,
   repeated_empty_dot_large_locals_deterministic=True,measurements=measured,
   deltas_per_iteration=deltas,extra_cost_per_callee=locals_costs,replies=saved,
   source_hashes={str(p.relative_to(ROOT)):sha(p)for p in dict.fromkeys(files)},
   scope='Installed local runtime metering. Compound operator bodies and per-callee local deltas only. No model inference, fee changes or goal completion claim.')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(extra_cost_per_callee=locals_costs,deltas=deltas)))
if __name__=='__main__':main()
