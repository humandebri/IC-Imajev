#!/usr/bin/env python3
"""Exact I32 balanced reductions: increase columns per accumulator update.
Sequential builds and local-only installs; frozen source/Wasm for every trial.
"""
import argparse,pathlib,subprocess,sys
ap=argparse.ArgumentParser();ap.add_argument('--canister',required=True);args=ap.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[1];source=ROOT/'crates/imajev-runtime/src/int8_kernel.rs';base=(ROOT/'artifacts/dot-unroll/baseline.rs').read_text()
def tree(expressions):
 if len(expressions)==1:return expressions[0]
 mid=len(expressions)//2;return f'i32x4_add({tree(expressions[:mid])},{tree(expressions[mid:])})'
for group in [8,16,32]:
 s=base.replace('step_by(32)',f'step_by({group*8})').replace('[[v128; 4]; C]',f'[[v128; {group}]; C]').replace('let input:[v128;4]',f'let input:[v128;{group}]')
 start=s.index('            let a=i32x4_add(');end=s.index('\n        })*};}',start)
 dots=[f'i32x4_dot_i16x8($input[{i}],weights[$j][{i}])' for i in range(group)]
 s=s[:start]+f'            acc[$i][$j]=i32x4_add(acc[$i][$j],{tree(dots)});'+s[end:];source.write_text(s)
 subprocess.run(['cargo','build','--release','--target','wasm32-unknown-unknown','-p','imajev-inference','--offline'],cwd=ROOT,check=True)
 wasm=ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm';name=f'dot-g{group}';frozen=ROOT/f'artifacts/dot-unroll/{name}.wasm';frozen.write_bytes(wasm.read_bytes());(ROOT/f'artifacts/dot-unroll/{name}.rs').write_text(s)
 subprocess.run(['icp','canister','install',args.canister,'--mode','upgrade','--wasm',str(frozen),'--network','local','--identity','imajev-local'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/profile_bottlenecks.py'),'--phase',name,'--source-directory','artifacts/bottleneck-fused-full-617','--normal-only','--canister',args.canister],cwd=ROOT,check=True)
 print('MEASURED',name,flush=True)
