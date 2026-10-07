#!/usr/bin/env python3
"""Compare current stack64 with stack64 spanning512 ascending columns."""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = ROOT / 'scripts/build_f32_output64_probe.py'
    source = original.read_text()
    source = source[source.index('def main():'):source.index("if __name__")]
    source = source.replace('artifacts/f32-output64-v1/build-v2', 'artifacts/f32-columns512-v1/build')
    source = source.replace("'rows%64==0'", "'rows%64==0&&cols%512==0'")
    a = "    (D/'src/lib.rs').write_text(lib)"
    assert source.count(a) == 1
    source = source.replace(a, """    lib=lib.replace('method<=1','method<=2')
    lib=lib.replace('else{f.packed.as_ref().unwrap().project64(&x,n).unwrap()}', 'else if method==1{f.packed.as_ref().unwrap().project64(&x,n).unwrap()}else{f.packed.as_ref().unwrap().project512(&x,n).unwrap()}')
    lib+='\\n#[ic_cdk::post_upgrade]fn post_upgrade(owner:candid::Principal,rows:u32,cols:u32){init(owner,rows,cols)}\\n'
    (D/'src/lib.rs').write_text(lib)""")
    a = "    (D/'src/packed.rs').write_text(packed[:packed.rfind('\\n}')] + '\\n' + wide + '\\n}\\n')"
    assert source.count(a) == 1
    source = source.replace(a, """    columns=wide.replace('pub fn project64(', 'pub fn project512(').replace('crate::kernel::accumulate64(', 'crate::kernel::accumulate512(')
    assert columns.count('for start in(0..self.cols).step_by(64)')==1
    columns=columns.replace('for start in(0..self.cols).step_by(64)', 'for start in(0..self.cols).step_by(512)')
    (D/'src/packed.rs').write_text(packed[:packed.rfind('\\n}')] + '\\n' + wide + '\\n' + columns + '\\n}\\n')""")
    a = "    (D/'src/kernel.rs').write_text(stub + '\\n' + second)"
    assert source.count(a) == 1
    source = source.replace(a, """    third=second.replace('__imajev_f32_wide','__imajev_f32_columns512').replace('fn accumulate64(', 'fn accumulate512(').replace('marker ^ 2','marker ^ 902')
    (D/'src/kernel.rs').write_text(stub + '\\n' + second + '\\n' + third)""")
    a = "    (D/'wide.wat').write_text(kernel64())"
    assert source.count(a) == 1
    source = source.replace(a, a+"\n    (D/'columns512.wat').write_text(kernel512())")
    source = source.replace("[D/'wide.wat', baseline, pathlib.Path(__file__)]", "[D/'wide.wat', D/'columns512.wat', baseline, pathlib.Path(__file__)]")
    source = source.replace("patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch'", "patcher = ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'")
    a = "(D/'control.wasm',D/'wide.wat',D/'diagnostic.wasm','__imajev_f32_wide')"
    assert source.count(a) == 1
    source = source.replace(a, "(D/'control.wasm',D/'wide.wat',D/'stack64.wasm','__imajev_f32_wide'),(D/'stack64.wasm',D/'columns512.wat',D/'diagnostic.wasm','__imajev_f32_columns512')")
    kernel_path = ROOT / 'scripts/build_f32_stack_probe.py'
    kernel_source = kernel_path.read_text().replace('width = 128', 'width = 64')
    ns = dict(__file__=__file__, __name__='columns512_generator')
    exec(compile(original.read_text(), str(original), 'exec'), ns)
    ns['__file__'] = __file__
    exec(compile(kernel_source, str(kernel_path), 'exec'), ns)
    kernel64 = ns['kernel']
    columns_source = kernel_source.replace('range(64)', 'range(512)').replace('__imajev_f32_wide', '__imajev_f32_columns512')
    ns512 = dict(__file__=__file__, __name__='columns512_generator')
    exec(compile(columns_source, str(kernel_path), 'exec'), ns512)
    wat = ns512['kernel']()
    count = len(re.findall(r'\(local \$', wat))
    assert count+9 <= 10000
    ns['kernel64'], ns['kernel512'] = kernel64, ns512['kernel']
    exec(compile(source, str(original), 'exec'), ns)
    ns['main']()
    d = ROOT / 'artifacts/f32-columns512-v1'
    (d/'frozen-builder.py').write_text(source)
    (d/'frozen-kernel-generator.py').write_text(columns_source)
    (d/'upstream.sha256').write_text(hashlib.sha256(original.read_bytes()).hexdigest()+'\n')
    print(f'columns512 kernel has {count} locals; all F32 columns ascending')


if __name__ == '__main__':
    main()
