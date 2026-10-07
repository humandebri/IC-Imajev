#!/usr/bin/env python3
"""Full inference common raw layout prerequisite; preserve complete rank7 tiling."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,a,b):
 assert s.count(a)==1,a
 return s.replace(a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume-patches',action='store_true');args=ap.parse_args()
 old=ROOT/'artifacts/update-k2-pair-guard-fold-v1/build';base=json.loads((old/'report.json').read_text())
 k=ROOT/'artifacts/k2-contiguous-roots-kernels-v1';kr=json.loads((k/'report.json').read_text());er=json.loads((k/'execution-report.json').read_text())
 assert er['complete'] and er['conditions']==208
 for r in [base,kr,er]:
  for key in ['source_hashes','dependency_hashes']:
   assert all(sha(ROOT/p)==h for p,h in r.get(key,{}).items())
 d=ROOT/'artifacts/update-common-raw-rank7-v1';d.mkdir(exist_ok=args.resume_patches);b=d/'build';b.mkdir(exist_ok=args.resume_patches)
 if not args.resume_patches:
  shutil.copytree(old/'runtime',b/'runtime')
  for p in old.glob('*.rs'):shutil.copyfile(p,b/p.name)
 else:
  assert (b/'raw.wasm').is_file() and not (b/'report.json').exists()
  for p in old.glob('*.rs'):assert p.read_bytes()==(b/p.name).read_bytes()
  for p in (old/'runtime').glob('*.rs'):
   if p.name!='strassen_raw.rs':assert p.read_bytes()==(b/'runtime'/p.name).read_bytes()
 p=b/'runtime/strassen_raw.rs';s=(old/'runtime/strassen_raw.rs').read_text()
 begin=s.index('pub(crate) fn index(');end=s.index('pub(crate) fn project(',begin)
 s=s[:begin]+'''pub(crate) fn index(rows:usize,cols:usize,row:usize,c:usize)->usize{
 ((c%256)/128*2+(row%8)/4)*(rows*cols/4)+(row/8)*cols*2+(c/256)*512+((c%128)/2)*8+(row%4)*2+c%2
}
pub(crate) fn pack(bytes:&[u8],rows:usize,cols:usize)->Vec<u8>{
 let mut data=vec![0;rows*cols];
 for row in 0..rows{for c in 0..cols{data[index(rows,cols,row,c)]=bytes[row*cols+c];}}
 data.extend_from_slice(&bytes[rows*cols..]);data
}
'''+s[end:]
 s=replace_once(s,'wp=core::array::from_fn(|m|fixed.data.as_ptr().add(m*stride+(start+r)/8*2*cols));','wp=core::array::from_fn(|m|fixed.data.as_ptr().add([0,2,1,3][m]*stride+(start+r)/8*2*cols));')
 s=replace_once(s,'wp=core::array::from_fn(|m|pad.as_ptr().add(m*(tile/4)*cols));','wp=core::array::from_fn(|m|pad.as_ptr().add([0,2,1,3][m]*(tile/4)*cols));')
 s=replace_once(s,'let p1=p.add(stride);let p2=p.add(2*stride);','let p1=p.add(2*stride);let p2=p.add(stride);')
 s=replace_once(s,'let lo=i32x4_shuffle::<0,4,1,5>(even,odd);let hi=i32x4_shuffle::<2,6,3,7>(even,odd);','let lo=even;let hi=odd;')
 if args.resume_patches:assert p.read_text()==s
 else:p.write_text(s)
 runtime=base['runtime_command'][:];runtime[runtime.index('--edition=2021')+1]=str(b/'runtime/lib.rs');runtime[runtime.index('-o')+1]=str(b/'libimajev_runtime.rlib')
 cmd=base['command'][:];cmd[cmd.index('--edition=2021')+1]=str(b/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm')
 for i,v in enumerate(cmd):
  if v.startswith('imajev_runtime='):cmd[i]='imajev_runtime='+str(b/'libimajev_runtime.rlib')
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(b),CARGO_PKG_NAME='imajev-runtime',CARGO_PKG_VERSION='0.1.0')
 if not args.resume_patches:
  with (b/'runtime-compiler.log').open('w') as log:subprocess.run(runtime,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 env.update(CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
 if not args.resume_patches:
  with (b/'compiler.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 paths={sha(ROOT/file):ROOT/file for file in base['source_hashes'] if file.endswith('.wat')}
 for p in list(old.parent.glob('*.wat'))+list(old.glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat']:paths[sha(p)]=p
 changed={v['symbol']:ROOT/v['path'] for v in kr['kernels']};assert len(changed)==8
 previous=b/'raw.wasm';patches=[];sources=[Path(__file__),old/'report.json',k/'report.json',k/'execution-report.json']+list(b.glob('*.rs'))+list((b/'runtime').glob('*.rs'))
 for i,patch in enumerate(base['patches']):
  if patch['export'] in changed:wat=changed[patch['export']]
  else:wat=paths[patch['source_sha256']];assert sha(wat)==patch['source_sha256']
  target=b/('full.wasm' if i==len(base['patches'])-1 else f'patched{i}.wasm')
  row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(wat),str(target),patch['export']],text=True));assert row['wasmparser_validation'];patches.append(row);sources.append(wat);previous=target
 assert len(patches)==30
 deps=[]
 for command in [runtime,cmd]:
  deps += [Path(command[i+1].split('=',1)[1]) for i,v in enumerate(command) if v=='--extern']
 sources=list(dict.fromkeys(sources));deps=list(dict.fromkeys(deps))
 report=dict(wasm_sha256=sha(previous),runtime_command=runtime,command=cmd,patches=patches,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in deps},rank7_tiles=[32,128,160,168],changed_kernel_bodies=8,unchanged_kernel_bodies=22,rank49_dispatch_included=False,scope='Full inference build: common four-plane I8 pack/index, rank7 pointer mapping/output roots, aligned SIMD one-token fallback. Current full tiling, padding, scalar readers and K-range code preserved. Node kernel proofs inherited; whole runtime/fidelity/performance not verified. No canister install/adoption.')
 (b/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(b/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in sources+[b/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(module=report['wasm_sha256'],bytes=previous.stat().st_size,validated_patches=30)))
if __name__=='__main__':main()
