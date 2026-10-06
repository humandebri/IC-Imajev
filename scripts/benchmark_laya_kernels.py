#!/usr/bin/env python3
"""Freeze real Laya weight slices and compare prepared dot/writeback in Wasm.

Laya checkout is read only. Output must be a fresh directory in this repository.
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

def sha(data): return hashlib.sha256(data).hexdigest()

def extract(source, declaration, last=False):
    start = source.rindex(declaration) if last else source.index(declaration)
    begin = source.index('{', start)
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--laya-root', default='/Volumes/KINGSTON/ICP/IC-Laya-Standalone')
    ap.add_argument('--directory', required=True)
    args = ap.parse_args()
    laya = Path(args.laya_root).resolve()
    dest = (ROOT / args.directory).resolve()
    if not dest.is_relative_to(ROOT): raise ValueError('output must stay in repository')
    dest.mkdir(parents=True, exist_ok=False)
    pack = laya / 'checkpoints/laya-int8'
    manifest_bytes = (pack / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['format'] == 'ic-laya-int8-pack-v1' and not manifest['test_only']
    source_path = laya / 'crates/laya-candle/src/int8.rs'
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode()
    # Exact current Wasm dot and scale/bias writeback; wrapper preserves q8 tile dispatch.
    dot = extract(source, 'fn dot_tile<const R: usize, const C: usize>(', last=True)
    store = extract(source, 'fn q8_store_tile<const R: usize, const C: usize>(', last=True)
    wrapper = '''
pub fn project(a: &[i8], sx: &[f32], w: &[i8], sw: &[f32], tokens: usize, rows: usize, cols: usize) -> Vec<f32> {
    assert!(a.len()==tokens*cols && w.len()==rows*cols && sx.len()==tokens && sw.len()==rows);
    let mut out=vec![0.;tokens*rows];
    let mut t=0;
    while t<tokens {
        macro_rules! tile {($r:literal)=>{{
            let mut r=0;
            while r<rows {
                let sums=dot_tile::<$r,16>(&a[t*cols..(t+$r)*cols],&w[r*cols..(r+16)*cols],cols);
                q8_store_tile::<$r,16>(&sums,&sx[t..t+$r],&sw[r..r+16],None,r,rows,&mut out[t*rows..(t+$r)*rows]);
                r+=16;
            }
            t+=$r;
        }};}
        let left=tokens-t;
        if left>=64{tile!(64);}else if left>=32{tile!(32);}else if left>=16{tile!(16);}
        else if left>=8{tile!(8);}else if left>=4{tile!(4);}else if left>=2{tile!(2);}else{tile!(1);}
    }
    out
}
'''
    (dest/'laya_extracted.rs').write_text('// Extracted from IC-Laya, MIT; see Laya-LICENSE.\n'+dot+'\n'+store+'\n'+wrapper)
    shutil.copyfile(laya/'LICENSE',dest/'Laya-LICENSE')
    tensors = []
    with (pack/'model.bin').open('rb') as model:
        for name, file, cols in [('encoder.0.qkv.weight','qkv.bin',1024),('encoder.0.wo.weight','wo.bin',2624)]:
            e = next(e for e in manifest['tensors'] if e['name']==name)
            rows, k = e['shape']
            assert rows>=512 and k==cols and e['storage']=='I8Row'
            assert e['length']==rows*cols+rows*4
            model.seek(e['offset']); data=model.read(e['length'])
            assert sha(data)==bytes(e['sha256']).hex(), name
            part=data[:512*cols]+data[rows*cols:rows*cols+512*4]
            (dest/file).write_bytes(part)
            tensors.append({'name':name,'shape':e['shape'],'row_start':0,'rows':512,'tensor_sha256':sha(data),'slice_sha256':sha(part),'file':file})
    manifest_path=ROOT/'scripts/laya_kernel_bench/Cargo.toml'
    subprocess.run(['cargo','generate-lockfile','--offline','--manifest-path',str(manifest_path)],cwd=ROOT,check=True)
    paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'crates/inference-core/src').rglob('*.rs'))
    paths+=list((ROOT/'scripts/laya_kernel_bench').rglob('*.rs'))
    paths += [ROOT/p for p in ['scripts/laya_kernel_bench/Cargo.toml','scripts/laya_kernel_bench/Cargo.lock','scripts/laya_kernel_bench/measure.mjs','scripts/benchmark_laya_kernels.py','scripts/measure_laya_kernels_ic.py','Cargo.toml','crates/imajev-runtime/Cargo.toml','crates/inference-core/Cargo.toml']]
    frozen={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    with zipfile.ZipFile(dest/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in frozen.items(): archive.writestr(name,data)
        archive.writestr('reference/laya-int8.rs',source_bytes)
        archive.writestr('reference/manifest.json',manifest_bytes)
        archive.writestr('generated/laya_extracted.rs',(dest/'laya_extracted.rs').read_bytes())
    report={'scope':'Real weight slices, synthetic prepared token-scaled activations; no quantization, output requantization, bias, model graph or communication in span',
        'laya_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=laya,text=True).strip(),
        'laya_source_sha256':sha(source_bytes),'manifest_sha256':sha(manifest_bytes),'model_revision':manifest['source_revision'],'tensors':tensors,
        'source_hashes':{name:sha(data) for name,data in frozen.items()},'source_archive_sha256':sha((dest/'source.zip').read_bytes())}
    (dest/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    env={**os.environ,'LAYA_KERNEL_FIXTURES':str(dest),'CARGO_TARGET_DIR':str(ROOT/'artifacts/runtime-comparison/laya-target')}
    cmd=['cargo','build','--offline','--locked','--release','--target','wasm32-unknown-unknown','--manifest-path',str(manifest_path)]
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    wasm=Path(env['CARGO_TARGET_DIR'])/'wasm32-unknown-unknown/release/laya_kernel_bench.wasm'
    shutil.copyfile(wasm,dest/'kernels.wasm')
    subprocess.run(['node','--no-liftoff',str(ROOT/'scripts/laya_kernel_bench/measure.mjs'),str(dest)],cwd=ROOT,check=True)
    for name,data in frozen.items(): assert (ROOT/name).read_bytes()==data, 'source changed: '+name
    assert source_path.read_bytes()==source_bytes and (pack/'manifest.json').read_bytes()==manifest_bytes
    report.update({'wasm_sha256':sha((dest/'kernels.wasm').read_bytes()),'build_command':cmd,'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'source_unchanged':True})
    (dest/'build.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
