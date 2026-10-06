#!/usr/bin/env python3
"""Compare pinned GGML Q8 Wasm dot and current Imajev prepared S1 projection.

Reads previously downloaded, hash-checked source. Builds standalone Wasm and
measures in Node; does not install/call a canister or download a model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def extract(source, declaration):
    start = source.index(declaration)
    begin = source.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--target-directory', help='Reuse a previous diagnostic Cargo cache')
    ap.add_argument('--local-ic', action='store_true', help='Include raw local-only IC diagnostic exports')
    args = ap.parse_args()
    dest = (ROOT / args.directory).resolve()
    if not dest.is_relative_to(ROOT):
        raise ValueError('output must stay inside the repository')
    dest.mkdir(parents=True, exist_ok=False)
    manifest = ROOT / 'scripts/runtime_kernel_bench/Cargo.toml'
    subprocess.run(['cargo', 'generate-lockfile', '--offline', '--manifest-path', str(manifest)], cwd=ROOT, check=True)
    paths = list((ROOT / 'crates/imajev-runtime/src').rglob('*.rs')) + list((ROOT / 'crates/inference-core/src').rglob('*.rs'))
    paths += list((ROOT / 'scripts/runtime_kernel_bench').rglob('*.rs'))
    paths += [ROOT / 'scripts/runtime_kernel_bench/Cargo.toml', ROOT / 'scripts/runtime_kernel_bench/Cargo.lock', ROOT / 'scripts/runtime_kernel_bench/measure.mjs', Path(__file__)]
    paths += [ROOT / 'Cargo.toml', ROOT / 'crates/imajev-runtime/Cargo.toml', ROOT / 'crates/inference-core/Cargo.toml']
    paths += [ROOT / p for p in ['artifacts/prefix_codec/full-wat/reuse.wat', 'artifacts/s1_address_reuse/build/kernel.wat', 'artifacts/s1_wide/build128/kernel.wat']]
    frozen = {str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    hashes = {name:hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
    with zipfile.ZipFile(dest / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in frozen.items(): archive.writestr(name, data)
    metadata = json.loads((ROOT / 'docs/runtime/comparison/fork-sources.json').read_text())
    fork = ROOT / 'artifacts/runtime-comparison/fork'
    for name, record in metadata['files'].items():
        assert hashlib.sha256((fork / name).read_bytes()).hexdigest() == record['sha256'], name
    quants = (fork / 'ggml/src/ggml-cpu/arch/wasm/quants.c').read_text()
    impl = (fork / 'ggml/src/ggml-impl.h').read_text()
    mappings = (fork / 'ggml/src/ggml-cpu/simd-mappings.h').read_text()
    functions = '\n'.join([
        extract(impl, 'static inline float fp32_from_bits('),
        extract(impl, 'static inline uint32_t fp32_to_bits('),
        extract(impl, 'static inline float ggml_compute_fp16_to_fp32('),
        extract(mappings, 'inline static float ggml_lookup_fp16_to_fp32('),
        extract(quants, 'void ggml_vec_dot_q8_0_q8_0('),
    ])
    header = '''// Generated from the pinned onicai/llama.cpp MIT source.
#include <stdint.h>
#include <stddef.h>
#include <wasm_simd128.h>
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)
#define assert(x) ((void)0)
#define memcpy __builtin_memcpy
#define QK8_0 32
typedef uint16_t ggml_fp16_t;
typedef struct { ggml_fp16_t d; int8_t qs[32]; } block_q8_0;
_Static_assert(sizeof(block_q8_0)==34, "Q8 block layout");
static float ggml_table_f32_f16[65536];
#define GGML_CPU_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
'''
    wrapper = '''
static block_q8_0 inputs[132*80], weights[512*80];
static float outputs[132*512];
static int tokens, rows, cols;
static int8_t value(uint32_t i, uint32_t seed) {return (int8_t)(((i*1664525u+seed)>>16)%63)-31;}
void setup(int n, int r, int c) {
  if (n<1 || n>132 || r!=512 || (c!=256 && c!=2560)) __builtin_trap();
  tokens=n; rows=r; cols=c;
  for (int i=0;i<65536;i++) ggml_table_f32_f16[i]=ggml_compute_fp16_to_fp32((uint16_t)i);
  for (int i=0;i<n*c/32;i++) {inputs[i].d=0x3c00;
    for (int j=0;j<32;j++) inputs[i].qs[j]=j==0?127:value((uint32_t)i*32+j,1013904223u);}
  for (int i=0;i<r*c/32;i++) {weights[i].d=0x3c00;
    for (int j=0;j<32;j++) weights[i].qs[j]=value((uint32_t)i*32+j,777u);}
}
void run(void) {
  for(int t=0;t<tokens;t++) for(int r=0;r<rows;r++)
    ggml_vec_dot_q8_0_q8_0(cols,&outputs[t*rows+r],0,&weights[r*(cols/32)],0,&inputs[t*(cols/32)],0,1);
}
uintptr_t output_ptr(void) {return (uintptr_t)outputs;}
int output_len(void) {return tokens*rows;}
#ifdef LOCAL_IC
__attribute__((import_module("ic0"),import_name("performance_counter"))) int64_t performance_counter(int32_t kind);
__attribute__((import_module("ic0"),import_name("msg_arg_data_size"))) int32_t msg_arg_data_size(void);
__attribute__((import_module("ic0"),import_name("msg_arg_data_copy"))) void msg_arg_data_copy(int32_t dst,int32_t offset,int32_t size);
__attribute__((import_module("ic0"),import_name("msg_reply_data_append"))) void msg_reply_data_append(int32_t src,int32_t size);
__attribute__((import_module("ic0"),import_name("msg_reply"))) void msg_reply(void);
__attribute__((export_name("canister_update configure"))) void configure(void) {
  if(msg_arg_data_size()!=12) __builtin_trap();
  uint32_t args[3]; msg_arg_data_copy((int32_t)(uintptr_t)args,0,12);
  setup(args[0],args[1],args[2]);run();msg_reply();
}
__attribute__((export_name("canister_query measure"))) void measure(void) {
  uint64_t begin=performance_counter(0);run();uint64_t instructions=performance_counter(0)-begin;
  uint64_t hash=14695981039346656037ull;
  const uint8_t *bytes=(const uint8_t*)outputs;
  for(int i=0;i<tokens*rows*4;i++) hash=(hash^bytes[i])*1099511628211ull;
  struct {uint64_t instructions,hash;uint32_t len,pages;} reply={instructions,hash,tokens*rows,__builtin_wasm_memory_size(0)};
  msg_reply_data_append((int32_t)(uintptr_t)&reply,sizeof(reply));msg_reply();
}
#endif
'''
    cfile = dest / 'ggml-harness.c'
    cfile.write_text(header + functions + wrapper)
    shutil.copyfile(fork / 'LICENSE', dest / 'llama.cpp-LICENSE')
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        subprocess.run(command, cwd=ROOT, check=True, **kwargs)

    run(['clang', '--target=wasm32', '-O3', '-msimd128', '-ffreestanding', '-DNDEBUG'] + (['-DLOCAL_IC'] if args.local_ic else []) + ['-c', str(cfile), '-o', str(dest / 'ggml.o')])
    sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
    host = next(line.split(': ', 1)[1] for line in subprocess.check_output(['rustc', '-vV'], text=True).splitlines() if line.startswith('host: '))
    linker = sysroot / 'lib/rustlib' / host / 'bin/rust-lld'
    run([str(linker), '-flavor', 'wasm', '--no-entry', '--export=setup', '--export=run', '--export=output_ptr', '--export=output_len', '--initial-memory=4194304'] + (['--allow-undefined'] if args.local_ic else []) + [str(dest / 'ggml.o'), '-o', str(dest / 'ggml.wasm')])
    target = (ROOT / args.target_directory).resolve() if args.target_directory else dest / 'target'
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    run(['cargo', 'build', '--offline', '--locked', '--release', '--target', 'wasm32-unknown-unknown', '--manifest-path', str(manifest)] + (['--features', 'local-ic'] if args.local_ic else []), env=env)
    module = target / 'wasm32-unknown-unknown/release/runtime_kernel_bench.wasm'
    patcher = ROOT / 'artifacts/wasm-audit-target/release/imajev-wasm-patch'
    patches = [
        ('__imajev_pair_accumulate', 'artifacts/prefix_codec/full-wat/reuse.wat'),
        ('__imajev_s1_raw_accumulate', 'artifacts/s1_address_reuse/build/kernel.wat'),
        ('__imajev_s1_wide_accumulate', 'artifacts/s1_wide/build128/kernel.wat'),
    ]
    for i, (export, source) in enumerate(patches):
        output = dest / f'imajev-patch-{i}.wasm'
        raw = subprocess.check_output([str(patcher), str(module), str(ROOT / source), str(output), export], text=True)
        (dest / f'patch-{i}.json').write_text(raw)
        module = output
    assert hashes == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in frozen}, 'source changed during build'
    run(['node', '--no-liftoff', str(ROOT / 'scripts/runtime_kernel_bench/measure.mjs'), str(module), str(dest / 'ggml.wasm'), str(dest / 'report.json')])
    assert hashes == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in frozen}, 'source changed during measurement'
    (dest / 'build.json').write_text(json.dumps({'commands':commands, 'clang':subprocess.check_output(['clang', '--version'], text=True), 'rustc':subprocess.check_output(['rustc', '-vV'], text=True), 'fork':metadata, 'source_hashes':hashes}, indent=2)+'\n')


if __name__ == '__main__':
    main()
