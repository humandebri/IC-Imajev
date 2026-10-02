#!/usr/bin/env python3
"""Sequential local-only builds/install/replays. Frozen Wasm per candidate."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];source=ROOT/'crates/imajev-runtime/src/int8_kernel.rs';base=(ROOT/'artifacts/bottleneck/scale-codec-int8.rs').read_text()
for limit in [8,16,32]:
 name=f'tile{limit}';s=base.replace('if t + 64 <= padded {',f'if {limit} >= 64 && t + 64 <= padded {{').replace('else if t + 32 <= padded {',f'else if {limit} >= 32 && t + 32 <= padded {{').replace('else if t + 16 <= padded {',f'else if {limit} >= 16 && t + 16 <= padded {{');source.write_text(s)
 subprocess.run(['cargo','build','--release','--target','wasm32-unknown-unknown','-p','imajev-inference','--offline'],cwd=ROOT,check=True)
 wasm=ROOT/'target/wasm32-unknown-unknown/release/imajev_inference.wasm';frozen=ROOT/f'artifacts/bottleneck/{name}.wasm';frozen.write_bytes(wasm.read_bytes());(ROOT/f'artifacts/bottleneck/{name}.rs').write_text(s)
 subprocess.run(['icp','canister','install','46el7-ql777-77775-aaada-cai','--mode','upgrade','--wasm',str(frozen),'--network','local','--identity','imajev-local'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/profile_bottlenecks.py'),'--phase',name,'--normal-only'],cwd=ROOT,check=True)
 print('MEASURED',name,hashlib.sha256(frozen.read_bytes()).hexdigest(),flush=True)
