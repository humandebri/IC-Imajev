#!/usr/bin/env python3
"""Compare exact integer token tiles and direct INT8 weight loading on a selected local canister."""
import argparse,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--canister',required=True);p.add_argument('--names',nargs='+',choices=['r32','r16','r8','c8','direct'],default=['r32','r16','r8','c8','direct']);args=p.parse_args()
source=ROOT/'crates/imajev-runtime/src/int8_kernel.rs';dest=ROOT/'artifacts/directions';base=(dest/'baseline.rs').read_text()
for name in args.names:
 s=base
 if name.startswith('r'):
  maximum=int(name[1:]);start=s.index('                    if t + 64 <= padded {');end=s.index('\n                };',start)
  sizes=[i for i in [64,32,16] if i<=maximum]
  branches=[]
  for i in sizes:branches.append(f'if t + {i} <= padded {{ project_tile::<{i}, $c>(q, &wide, scales, rows, t, r, &mut out); t += {i}; }} else ')
  s=s[:start]+'                    '+''.join(branches)+'{ project_tile::<8, $c>(q, &wide, scales, rows, t, r, &mut out); t += 8; }'+s[end:]
 if name=='c8':
  start=s.index('            if r + 16 <= rows {');end=s.index('\n    }\n    if !out.iter()',start)
  s=s[:start]+'            dispatch!(8);\n        }\n        r += 8;'+s[end:]
 if name=='direct':
  s=s.replace('let wide = crate::profile::measure("weight_widen", || widen(w));','let wide = w;')
  s=s.replace('wide: &[i16]','wide: &[i8]').replace('w: *const i16','w: *const i8').replace('let wp: [*const i16; C]','let wp: [*const i8; C]')
  s=s.replace('v128_load(wp[j].add(c + g * 8).cast())','i16x8_extend_low_i8x16(v128_load64_zero(wp[j].add(c + g * 8).cast()))')
  a=s.index('fn widen(w: &[i8])');b=s.index('\n#[cfg(test)]',a);s=s[:a]+s[b:]
 source.write_text(s)
 subprocess.run(['cargo','build','--release','--target','wasm32-unknown-unknown','-p','imajev-inference','--offline'],cwd=ROOT,check=True)
 wasm=ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm';frozen=dest/f'{name}.wasm';frozen.write_bytes(wasm.read_bytes());(dest/f'{name}.rs').write_text(s)
 subprocess.run(['icp','canister','install',args.canister,'--mode','upgrade','--wasm',str(frozen),'--network','local','--identity','imajev-local'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/profile_bottlenecks.py'),'--phase','directions-'+name,'--source-directory','artifacts/mlp-wide-full-617','--normal-only','--integer-only','--canister',args.canister],cwd=ROOT,check=True)
 print('MEASURED',name,flush=True)
