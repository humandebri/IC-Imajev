#!/usr/bin/env python3
"""Build exact output128 F32 probe sharing original packed output32 weights."""
import pathlib
import build_f32_output64_probe as original

def kernel():
 width=128;vectors=width//4
 lines=['(module (func (export "__imajev_f32_wide") '+' '.join(f'(param ${p} i32)' for p in ['q','w','cols','start','sx','stride','sw','sums','n']), '(local $t i32) (local $qp i32) (local $yp i32) (local $x v128)']
 lines += [f'(local $wp{p} i32)' for p in range(width//32)]+[f'(local $s{r} v128)' for r in range(vectors)]+[f'(local $w{c}_{r} v128)' for c in range(64) for r in range(vectors)]
 def token(first):
  out=[]
  for p in range(width//32):out += [f'(local.set $wp{p} (i32.add (local.get $w) (i32.shl (i32.add (local.get $start) (i32.mul (local.get $cols) (i32.const {p}))) (i32.const 7))))']
  out += ['(local.set $qp (i32.add (local.get $q) (i32.shl (i32.add (i32.mul (local.get $t) (local.get $cols)) (local.get $start)) (i32.const 2))))','(local.set $yp (i32.add (local.get $sums) (i32.shl (i32.mul (local.get $t) (local.get $stride)) (i32.const 2))))']
  out += [f'(local.set $s{r} (v128.load offset={r*16} (local.get $yp)))' for r in range(vectors)]
  for c in range(64):
   for r in range(vectors):
    x=f'(local.tee $x (v128.load32_splat offset={c*4} (local.get $qp)))' if r==0 else '(local.get $x)'
    w=f'(local.tee $w{c}_{r} (v128.load offset={c*128+(r%8)*16} (local.get $wp{r//8})))' if first else f'(local.get $w{c}_{r})'
    out += [f'(local.set $s{r} (f32x4.add (local.get $s{r}) (f32x4.mul {x} {w})))']
  out += [f'(v128.store offset={r*16} (local.get $yp) (local.get $s{r}))' for r in range(vectors)]
  return out
 lines += ['(local.set $t (i32.const 0))','(block $done (br_if $done (i32.eqz (local.get $n)))']+token(True)+['(local.set $t (i32.const 1))','(loop $tokens (br_if $done (i32.ge_u (local.get $t) (local.get $n)))']+token(False)+['(local.set $t (i32.add (local.get $t) (i32.const 1)))','br $tokens','))','))']
 return '\n'.join(lines)+'\n'

def main():
 source=pathlib.Path(original.__file__).read_text();source=source[source.index('def main():'):source.index("if __name__")]
 source=source.replace('artifacts/f32-output64-v1/build-v2','artifacts/f32-output128-v1/build').replace('f32_output64_probe','f32_output128_probe').replace("'rows%64==0'","'rows%128==0'")
 source=source.replace("(D/'src/lib.rs').write_text(lib)", "lib+='\\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:candid::Principal,rows:u32,cols:u32){init(owner,rows,cols)}\\n'\n    (D/'src/lib.rs').write_text(lib)")
 source=source.replace("'step_by(64)'","'step_by(128)'").replace("'[[0f32;64]'","'[[0f32;128]'").replace("'x.as_ptr(),64,'","'x.as_ptr(),128,'").replace("'r+64]'","'r+128]'").replace("'n*64'","'n*128'")
 source=source.replace("patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'","patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'")
 ns=dict(original.__dict__);ns['__file__']=__file__;ns['kernel64']=kernel
 exec(source,ns);ns['main']()
if __name__=='__main__':main()
