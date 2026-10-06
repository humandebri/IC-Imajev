#!/usr/bin/env python3
"""Build an exact key-major Delta diagnostic with register-resident state."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/delta-register-v3/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False)
 lines=['(module (memory 1) (func (export "__imajev_delta_register") '+ ' '.join(f'(param ${p} i32)' for p in ['q','k','v','g','b','state','out','unused','n'])]
 lines+=['(local $t i32) (local $qp i32) (local $kp i32) (local $vp i32) (local $op i32) (local $decay v128) (local $beta v128) (local $ki v128) (local $qi v128)']
 lines += [f'(local $s{i}_{j} v128)' for i in range(128) for j in range(32)]
 lines += [f'(local $m{j} v128) (local $u{j} v128) (local $o{j} v128)' for j in range(32)]
 for i in range(128):
  for j in range(32):lines+=[f'(local.set $s{i}_{j} (v128.load offset={(i*128+j*4)*4} (local.get $state)))']
 lines+=['(block $done (loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']
 for p in ['q','k','v','out']:lines += [f'(local.set ${"op" if p=="out" else p+"p"} (i32.add (local.get ${p}) (i32.shl (local.get $t) (i32.const 9))))']
 for p,l in [('g','decay'),('b','beta')]:lines += [f'(local.set ${l} (v128.load32_splat (i32.add (local.get ${p}) (i32.shl (local.get $t) (i32.const 2)))))']
 for j in range(32):lines += [f'(local.set $m{j} (v128.const i32x4 0 0 0 0))']
 for i in range(128):
  lines += [f'(local.set $ki (v128.load32_splat offset={i*4} (local.get $kp)))']
  for j in range(32):lines += [f'(local.set $s{i}_{j} (f32x4.mul (local.get $s{i}_{j}) (local.get $decay)))',f'(local.set $m{j} (f32x4.add (local.get $m{j}) (f32x4.mul (local.get $s{i}_{j}) (local.get $ki))))']
 for j in range(32):lines += [f'(local.set $u{j} (f32x4.mul (f32x4.sub (v128.load offset={j*16} (local.get $vp)) (local.get $m{j})) (local.get $beta)))',f'(local.set $o{j} (v128.const i32x4 0 0 0 0))']
 for i in range(128):
  for p in ['k','q']:lines += [f'(local.set ${p}i (v128.load32_splat offset={i*4} (local.get ${p}p)))']
  for j in range(32):lines += [f'(local.set $s{i}_{j} (f32x4.add (local.get $s{i}_{j}) (f32x4.mul (local.get $ki) (local.get $u{j}))))',f'(local.set $o{j} (f32x4.add (local.get $o{j}) (f32x4.mul (local.get $s{i}_{j}) (local.get $qi))))']
 for j in range(32):lines += [f'(v128.store offset={j*16} (local.get $op) (local.get $o{j}))']
 lines+=['(local.set $t (i32.add (local.get $t) (i32.const 1))) (br $tokens)))']
 for i in range(128):
  for j in range(32):lines+=[f'(v128.store offset={(i*128+j*4)*4} (local.get $state) (local.get $s{i}_{j}))']
 lines+=['))']
 (D/'kernel.wat').write_text('\n'.join(lines)+'\n')
 source=(ROOT/'scripts/delta_writeback_bench/src/lib.rs').read_text()
 source=source[:source.index('#[cfg(test)]')]
 source=source.replace('#[path="../../../crates/imajev-runtime/src/delta_simd.rs"]','#[path="delta_simd.rs"]')
 start=source.index(' #[cfg(target_arch="wasm32")]\n let out=unsafe {if key_major')
 end=source.index(' #[cfg(not(target_arch="wasm32"))]',start)
 source=source[:start]+''' #[cfg(target_arch="wasm32")]
 let out=unsafe {if key_major {let mut out=vec![0.;n as usize*128];__imajev_delta_register(q.as_ptr()as usize,k.as_ptr()as usize,v.as_ptr()as usize,g.as_ptr()as usize,b.as_ptr()as usize,state.as_mut_ptr()as usize,out.as_mut_ptr()as usize,0,n as usize);out}else{delta_simd::run::<false,true>(&q,&k,&v,&g,&b,&mut state,128,128)}};
'''+source[end:]
 # Both candidates get identical key-major initial state, conversion outside kernel.
 source=source.replace('if key_major {let old=state.clone();','{let old=state.clone();')
 source=source.replace('assert!((1..=132).contains(&n)&&dk>0&&dk<=128&&dv>0&&dv<=128&&dv%4==0);','assert!((1..=132).contains(&n)&&dk==128&&dv==128);')
 source=source.replace('for v in &out {hash.update(v.to_le_bytes());}', 'for v in out.iter().chain(state.iter()) {hash.update(v.to_le_bytes());}')
 source+='''
#[unsafe(no_mangle)] #[inline(never)]
pub unsafe extern "C" fn __imajev_delta_register(q:usize,k:usize,v:usize,g:usize,b:usize,state:usize,out:usize,unused:usize,n:usize) {core::hint::black_box((q,k,v,g,b,state,unused,n));(out as *mut f32).write(f32::NAN);}
'''
 (D/'lib.rs').write_text(source)
 frozen=ROOT/'artifacts/update-inference/source-v2/crates/imajev-runtime/src/delta_simd.rs'
 (D/'delta_simd.rs').write_bytes(frozen.read_bytes())
 old=json.loads((ROOT/'artifacts/s1_wide/raw128/report.json').read_text())
 command=old['command'][:];command[command.index('--edition=2021')+1]=str(D/'lib.rs');command[command.index('-o')+1]=str(D/'raw.wasm')
 sources=[D/'lib.rs',D/'delta_simd.rs',D/'kernel.wat',Path(__file__),frozen]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
 for p,h in old['dependency_hashes'].items():assert sha(ROOT/p)==h
 with (D/'compiler.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=dict(os.environ,**old['explicit_env']),stdout=log,stderr=log,check=True)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'
 patch=json.loads(subprocess.check_output([str(patcher),str(D/'raw.wasm'),str(D/'kernel.wat'),str(D/'diagnostic.wasm'),'__imajev_delta_register'],text=True))
 assert hashes=={p:sha(ROOT/p) for p in hashes}
 (D/'report.json').write_text(json.dumps(dict(command=command,source_hashes=hashes,dependency_hashes=old['dependency_hashes'],patch=patch,wasm_sha256=sha(D/'diagnostic.wasm'),scope=__doc__),indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(patch))
if __name__=='__main__':main()
