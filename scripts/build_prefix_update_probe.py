#!/usr/bin/env python3
"""Build both exact prefix restoration variants without installing canisters."""
import argparse, hashlib, json, pathlib, shutil, subprocess, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--directory',required=True)
ap.add_argument('--target-directory',required=True)
a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
assert not (d/'report.json').exists() and not (d/'source.zip').exists()
if not (ROOT/'scripts/prefix_update_bench/Cargo.lock').exists():
 subprocess.run(['cargo','generate-lockfile','--offline','--manifest-path','scripts/prefix_update_bench/Cargo.toml'],cwd=ROOT,check=True)
paths=list((ROOT/'crates/imajev-runtime/src').rglob('*.rs'))+list((ROOT/'scripts/prefix_update_bench/src').rglob('*.rs'))
paths += [ROOT/'crates/imajev-runtime/Cargo.toml',ROOT/'scripts/prefix_update_bench/Cargo.toml',ROOT/'scripts/prefix_update_bench/Cargo.lock',pathlib.Path(__file__)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={str(p.relative_to(ROOT)):sha(p)for p in paths}
with zipfile.ZipFile(d/'source.zip','x',zipfile.ZIP_DEFLATED)as z:
 for p in paths:z.write(p,str(p.relative_to(ROOT)))
rows=[]
for variant,features in [('baseline',[]),('hoisted',['experimental-prefix-update-hoist'])]:
 cmd=['cargo','build','--offline','--release','-j1','--manifest-path','scripts/prefix_update_bench/Cargo.toml','--target','wasm32-unknown-unknown','--target-dir',a.target_directory,'--lib']
 if features:cmd+=['--features',','.join(features)]
 with (d/f'{variant}.log').open('x')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 out=d/f'{variant}.wasm';shutil.copyfile(ROOT/a.target_directory/'wasm32-unknown-unknown/release/imajev_prefix_update_bench.wasm',out)
 rows.append(dict(variant=variant,command=cmd,wasm_sha256=sha(out),wasm_bytes=out.stat().st_size));print(json.dumps(rows[-1]),flush=True)
assert sources=={str(p.relative_to(ROOT)):sha(p)for p in paths}
(d/'report.json').write_text(json.dumps(dict(source_hashes=sources,builds=rows,installed=False),indent=2)+'\n')
