#!/usr/bin/env python3
"""Build the frozen query32 kernels with a pinned merged-pack execution mode.

The protocol keeps rank64 operand slots, filled with zero when no adapter exists.
Internal zero descriptors are never uploaded or read from stable memory. All A/B
matrix products and column continuations are skipped for the pinned merged pack.
"""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/voting-template-prefix-v1/full-build'
D=ROOT/'artifacts/merged-query32-v1/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(p,old,new):
 s=p.read_text();assert s.count(old)==1,(p,old,s.count(old));p.write_text(s.replace(old,new))
def main():
 D.mkdir(parents=True,exist_ok=False)
 report=json.loads((BASE/'report.json').read_text())
 assert sha(BASE/'full.wasm')==report['wasm_sha256']
 shutil.copytree(BASE/'runtime',D/'runtime')
 for p in BASE.glob('*.rs'):shutil.copyfile(p,D/p.name)
 original=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text())
 merged=json.loads((ROOT/'artifacts/merged-adapter-v2/model.manifest.json').read_text())
 specs=[t for t in original['tensors'] if '.lora_' in t['name']]
 assert len(specs)==400 and not any('.lora_' in t['name'] for t in merged['tensors'])
 # Exact hash binding prevents absent or partially supplied adapters in any
 # other model from being silently interpreted as a merged model.
 helper='''\npub fn merged_execution_manifest(m:&Manifest)->Option<Manifest> {
 if m.pack_hash != "'''+merged['pack_hash']+'''" { return None; }
 let mut out=m.clone();
 if out.tensors.iter().any(|t|t.name.contains(".lora_")) { return None; }
'''
 for t in specs:
  helper+=f' out.tensors.push(Tensor{{name:"{t["name"]}".into(),offset:u64::MAX,rows:{t["rows"]},cols:{t["cols"]},dtype:"f32".into(),bytes:{t["bytes"]}}});\n'
 helper+=' Some(out)\n}\n'
 p=D/'runtime/lib.rs';p.write_text(p.read_text()+helper)
 replace(p,'    let count = rows.checked_mul(t.cols).ok_or("tile overflow")?;','''    if t.offset==u64::MAX && t.name.contains(".lora_") && t.dtype=="f32" {
        return Ok((LoadedWeight::Prepared(prepared_weights::PreparedF32::absent_adapter(rows*t.cols)),0));
    }
    let count = rows.checked_mul(t.cols).ok_or("tile overflow")?;''')
 replace(p,'    let out = if let Some((a, b)) = lora {','''    let out = if lora.is_some_and(|(a,b)|a.offset==u64::MAX && b.offset==u64::MAX) {
        base.into_iter().map(bf).collect()
    } else if let Some((a, b)) = lora {''')
 p=D/'runtime/prepared_weights.rs'
 replace(p,'impl PreparedF32 {','''impl PreparedF32 {
    pub(crate) fn absent_adapter(len:usize)->Self {
        Self {values:Rc::from([]),start:0,len,output_rows:None,output_cols:64,original:Rc::new(OnceCell::new())}
    }
    pub(crate) fn is_absent_adapter(&self)->bool {self.values.is_empty() && self.len>0}
''')
 p=D/'runtime/f32_output.rs'
 replace(p,'    let (packed, start, prepared_cols) = w.output_view()?;','''    if w.is_absent_adapter() {
        return Some(if n>0 && rows*cols==w.len() && x.len()==n*cols && x.iter().all(|v|v.is_finite()) {Ok(vec![0.;n*rows])} else {Err("absent adapter projection shape".into())});
    }
    let (packed, start, prepared_cols) = w.output_view()?;''')
 replace(p,' let(packed,start,prepared_cols)=w.output_view().ok_or("F32 continuation layout")?;',''' if w.is_absent_adapter() {
  if n==0 || initial.len()!=n*rows || x.len()!=n*count || rows*cols!=w.len() || begin.checked_add(count).is_none_or(|end|end>cols) || !initial.iter().chain(x).all(|v|v.is_finite()) {return Err("absent adapter continuation shape".into());}
  return Ok(initial.to_vec());
 }
 let(packed,start,prepared_cols)=w.output_view().ok_or("F32 continuation layout")?;''')
 p=D/'runtime/int8_k_continue.rs'
 replace(p,'        let mut br = r.clone();','''        if a.offset==u64::MAX && b.offset==u64::MAX {
            return Ok((x[..n*rows].iter().map(|v|crate::bf(*v)).collect(),0));
        }
        let mut br = r.clone();''')
 p=D/'lib.rs';s=p.read_text()
 for needle in ['        let m=s.manifest.as_ref().ok_or("missing manifest")?;','        let m = s.manifest.as_ref().ok_or("missing manifest")?;']:
  assert s.count(needle)==1
  s=s.replace(needle,needle+'\n        let merged=imajev_runtime::merged_execution_manifest(m);\n        let m=merged.as_ref().unwrap_or(m);')
 p.write_text(s)
 def remap(command):
  return [str(arg).replace(str(BASE),str(D)) for arg in command]
 runtime=remap(report['runtime_command']);wrapper=remap(report['canister_command'])
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(D),CARGO_PKG_NAME='imajev-runtime',CARGO_PKG_VERSION='0.1.0')
 with (D/'runtime-compiler.log').open('w') as log:subprocess.run(runtime,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 env.update(CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
 with (D/'compiler.log').open('w') as log:subprocess.run(wrapper,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';previous=D/'raw.wasm';patches=[]
 for i,old in enumerate(report['patches']):
  # Read the recorded kernel path; retain the exact same six numerical kernels.
  wat=Path(old['wat']) if 'wat' in old else None
  if wat is None:
   if i<4:wat=ROOT/'artifacts/single-quad/build-v2'/f'kernel{i}.wat'
   if i==3:wat=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
   if i==4:wat=ROOT/'artifacts/delta-register-v3/build/kernel.wat'
   if i==5:wat=ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat'
  output=D/('full.wasm' if i==5 else f'patched{i}.wasm')
  result=json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(output),old['export']],text=True));assert result['wasmparser_validation'];patches.append(result);previous=output
 sources={str(p.relative_to(ROOT)):sha(p) for p in list(D.glob('*.rs'))+list((D/'runtime').glob('*.rs'))+[Path(__file__)]}
 (D/'report.json').write_text(json.dumps(dict(baseline=report['wasm_sha256'],wasm_sha256=sha(D/'full.wasm'),merged_pack_hash=merged['pack_hash'],zero_operand_slots=64,scope=__doc__,source_hashes=sources,patches=patches,runtime_command=runtime,canister_command=wrapper),indent=2)+'\n')
 print(json.dumps(dict(wasm_sha256=sha(D/'full.wasm'))),flush=True)
if __name__=='__main__':main()
