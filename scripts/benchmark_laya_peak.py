#!/usr/bin/env python3
"""Test Imajev's unsigned-bit peak scan on a frozen copy of Laya quantization."""
import argparse, hashlib, json, os, shutil, subprocess, zipfile
from pathlib import Path
from repository_paths import existing_directory
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);ap.add_argument('--laya-root',type=existing_directory, required=True);args=ap.parse_args()
    dest=(ROOT/args.directory).resolve();laya=Path(args.laya_root).resolve()
    if not dest.is_relative_to(ROOT):raise ValueError('output must stay inside repository')
    dest.mkdir(parents=True,exist_ok=False)
    source=(laya/'crates/laya-candle/src/int8_quant.rs').read_bytes();baseline='\n'.join(source.decode().splitlines()[1:])+'\n'
    start=baseline.index('        let mut peak = f32x4_splat(0.);')
    end=baseline.index('        let scale = if peak == 0.',start)
    replacement='''        let mask = i32x4_splat(0x7fffffff);
        let mut peak = i32x4_splat(0);
        for i in (0..end).step_by(4) {
            peak = u32x4_max(peak, v128_and(unsafe { v128_load(input.as_ptr().add(i).cast()) }, mask));
        }
        peak = u32x4_max(peak, i32x4_shuffle::<2,3,0,1>(peak,peak));
        peak = u32x4_max(peak, i32x4_shuffle::<1,0,3,2>(peak,peak));
        let mut bits = u32x4_extract_lane::<0>(peak);
        for value in &input[end..] { bits = bits.max(value.to_bits() & 0x7fffffff); }
        if bits >= 0x7f800000 { candle_core::bail!("nonfinite int8 activation") }
        let peak = f32::from_bits(bits);
'''
    candidate=baseline[:start]+replacement+baseline[end:]
    (dest/'baseline.rs').write_text(baseline);(dest/'candidate.rs').write_text(candidate);shutil.copyfile(laya/'LICENSE',dest/'Laya-LICENSE')
    manifest=ROOT/'scripts/laya_peak_bench/Cargo.toml';subprocess.run(['cargo','generate-lockfile','--offline','--manifest-path',str(manifest)],cwd=ROOT,check=True)
    paths=list((ROOT/'scripts/laya_peak_bench/src').rglob('*.rs'))+[ROOT/p for p in ['scripts/laya_peak_bench/Cargo.toml','scripts/laya_peak_bench/Cargo.lock','scripts/laya_peak_bench/measure.mjs','scripts/benchmark_laya_peak.py','scripts/measure_laya_peak_ic.py','crates/imajev-runtime/src/quantize_simd.rs']]
    frozen={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    with zipfile.ZipFile(dest/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,b in frozen.items():z.writestr(name,b)
        z.writestr('reference/laya-int8_quant.rs',source);z.writestr('generated/baseline.rs',baseline);z.writestr('generated/candidate.rs',candidate)
    env={**os.environ,'LAYA_PEAK_SOURCE':str(dest),'CARGO_TARGET_DIR':str(ROOT/'artifacts/runtime-comparison/laya-peak-target')}
    subprocess.run(['cargo','build','--offline','--locked','--release','--target','wasm32-unknown-unknown','--manifest-path',str(manifest)],cwd=ROOT,env=env,check=True)
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'wasm32-unknown-unknown/release/laya_peak_bench.wasm',dest/'diagnostic.wasm')
    subprocess.run(['node',str(ROOT/'scripts/laya_peak_bench/measure.mjs'),str(dest)],cwd=ROOT,check=True)
    for name,b in frozen.items():assert (ROOT/name).read_bytes()==b,name
    assert (laya/'crates/laya-candle/src/int8_quant.rs').read_bytes()==source
    report={'scope':'Laya row quantization with allocation and diagnostic output assembly, not full inference',
        'laya_source_sha256':sha(source),'baseline_sha256':sha(baseline.encode()),'candidate_sha256':sha(candidate.encode()),'wasm_sha256':sha((dest/'diagnostic.wasm').read_bytes()),'source_archive_sha256':sha((dest/'source.zip').read_bytes()),'source_hashes':{name:sha(b) for name,b in frozen.items()},'source_unchanged':True,'rustc':subprocess.check_output(['rustc','--version'],text=True).strip()}
    (dest/'build.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
