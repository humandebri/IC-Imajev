#!/usr/bin/env python3
"""Diagnostic guarded release runtime: checked external pair products and opaque patched-kernel sentinels; canister overflow checks retained."""
import hashlib,json,os,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/voting-template-prefix-v1/full-build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=False)
 baseline=ROOT/'artifacts/query-packing-v3/build';provenance=json.loads((baseline/'provenance.json').read_text())
 for p,h in provenance['frozen_runtime_source_sha256'].items():assert sha(ROOT/p)==h
 for p,h in provenance['extern_rlibs_sha256'].items():assert sha(ROOT/p)==h
 src=ROOT/'artifacts/update-inference/source-v2/crates/imajev-runtime/src';shutil.copytree(src,D/'runtime')
 p=D/'runtime/delta_simd.rs';s=p.read_text();needle='    if dv < 128 {';assert s.count(needle)==1
 s=s.replace(needle,'''    if KEY_MAJOR && !WRITEBACK && dk==128 && dv==128 {
        let mut out=vec![0.;g.len()*128];
        __imajev_delta_register(q.as_ptr()as usize,k.as_ptr()as usize,v.as_ptr()as usize,g.as_ptr()as usize,beta.as_ptr()as usize,state.as_mut_ptr()as usize,out.as_mut_ptr()as usize,0,g.len());
        return out;
    }
'''+needle,1)
 stub=(ROOT/'artifacts/delta-register-v3/build/lib.rs').read_text().split('#[unsafe(no_mangle)]')[-1];s+='\n#[cfg(target_arch="wasm32")]\n#[unsafe(no_mangle)]'+stub;p.write_text(s)
 # Add exact-content fixed-prefix caching to the frozen runtime.
 shutil.copyfile(ROOT/'scripts/prefix_state_cache.rs',D/'runtime/prefix_state_cache.rs')
 p=D/'runtime/lib.rs';s=p.read_text();s+='\n#[cfg(feature="experimental-delta-state-layout")]pub mod prefix_state_cache;\n';p.write_text(s)
 p=D/'runtime/delta_restore.rs';s=p.read_text();needle='pub(crate) fn restore_key_major(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> {crate'
 # Frozen source delegates directly to restore_layout, retain fallback exactly.
 needle='pub(crate) fn restore_key_major(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> { restore_layout::<true>(n,h,x) }'
 assert needle in s
 s=s.replace(needle,'pub(crate) fn restore_key_major(n:usize,h:usize,x:&[f32])->Result<Vec<f32>> { if let Some(state)=crate::prefix_state_cache::lookup_log(n,h,x){return Ok(state);}restore_layout::<true>(n,h,x) }');p.write_text(s)
 p=D/'runtime/prefix_hybrid_codec.rs';s=p.read_text();needle='{Ok(crate::delta_full_log::InitialState::KeyMajor(decode_layout::<true>(bytes)?))}'
 assert needle in s
 s=s.replace(needle,'{if let Some(state)=crate::prefix_state_cache::lookup_packet(bytes){return Ok(crate::delta_full_log::InitialState::KeyMajor(state));}Ok(crate::delta_full_log::InitialState::KeyMajor(decode_layout::<true>(bytes)?))}');p.write_text(s)
 for p in baseline.glob('*.rs'):shutil.copyfile(p,D/p.name)
 shutil.copyfile(ROOT/'scripts/prefix_state_cache_canister.rs',D/'prefix_state_cache_canister.rs')
 p=D/'lib.rs';s=p.read_text();s+='\nmod prefix_state_cache_canister;\n';p.write_text(s)
 shutil.copyfile(ROOT/'scripts/query_attention_mlp_front.rs',D/'query_attention_mlp_front.rs')
 p=D/'lib.rs';s=p.read_text();s+='\nmod query_attention_mlp_front;\n';p.write_text(s)
 # Keep externally supplied shapes checked before runtime index arithmetic.
 p=D/'runtime/lib.rs';s=p.read_text();needle='    let payload = &b[4 + n..b.len() - 32];'
 assert s.count(needle)==1
 s=s.replace(needle,"""    // Pair products guard external dimensions against Wasm32 wraparound.
    // Canonical model operations additionally enforce their fixed shape caps.
    for (i, &a) in r.dims.iter().enumerate() {
        for &b in &r.dims[i+1..] {
            if a.checked_mul(b).is_none() { return Err(\"shape product overflow\".into()); }
        }
    }
"""+needle);p.write_text(s)
 # Opaque fail-closed values prevent LTO inferring NaN at callers of patched ABI.
 for name in ['output_pairs_wat.rs','strassen_raw.rs','f32_output.rs','delta_simd.rs']:
  p=D/'runtime'/name;s=p.read_text()
  s=s.replace('f32::from_bits(marker|0x7fc00000)','core::hint::black_box(f32::from_bits(marker|0x7fc00000))')
  s=s.replace('f32::from_bits(marker | 0x7fc00000)','core::hint::black_box(f32::from_bits(marker | 0x7fc00000))')
  s=s.replace('.write(f32::NAN)', '.write(core::hint::black_box(f32::NAN))')
  if name=='output_pairs_wat.rs':s=s.replace('marker|0x7fc00000','(marker^1)|0x7fc00000')
  if name=='strassen_raw.rs':
   s=s.replace('marker|0x7fc00000','(marker^2)|0x7fc00000',1)
   s=s.replace('marker|0x7fc00000','(marker^4)|0x7fc00000',1)
  if name=='f32_output.rs':s=s.replace('marker | 0x7fc00000','(marker ^ 3) | 0x7fc00000')
  p.write_text(s)
 p=D/'query_mlp_delta.rs';s=p.read_text();assert '1..=27' in s;s=s.replace('1..=27','1..=38');p.write_text(s)
 p=D/'runtime/f32_output.rs';s=p.read_text();needle='        let mut out = vec![0.; n * rows];';assert s.count(needle)==1
 s=s.replace(needle,'        #[cfg(target_arch="wasm32")]\n        if rows%64==0 && start%32==0 && cols%64==0 {return project_wide(x,packed,start,n,rows,cols); }\n'+needle)
 s+='\n'+(ROOT/'scripts/f32_output64_full.rs').read_text();p.write_text(s)
 deps=ROOT/'artifacts/update-inference/target/wasm32-unknown-unknown/release/deps'
 runtime_fp=json.loads((deps.parent/'.fingerprint/imajev-runtime-1aa8c9cafc9b14a6/lib-imajev_runtime.json').read_text())
 cmd=['rustc','--crate-name','imajev_runtime','--edition=2021',str(D/'runtime/lib.rs'),'--crate-type','rlib','--target','wasm32-unknown-unknown','-C','opt-level=3','-C','panic=abort','-C','codegen-units=1','-C','embed-bitcode=yes','-C','overflow-checks=no','-C','metadata=voting_template_prefix_v1','-o',str(D/'libimajev_runtime_register.rlib'),'-L','dependency='+str(deps),'-L','dependency='+str(ROOT/'artifacts/update-inference/target/release/deps')]
 hashes={}
 for name in ['inference_core','serde','serde_json','sha2','blake3','miniz_oxide']:
  choices=list(deps.glob('lib'+name+'-*.rlib'));assert len(choices)==1,(name,choices);cmd+=['--extern',name+'='+str(choices[0])];hashes[str(choices[0].relative_to(ROOT))]=sha(choices[0])
 for f in json.loads(runtime_fp['features']):cmd+=['--cfg','feature="'+f+'"']
 sources=[ROOT/'scripts/f32_output64_full.rs',ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat',ROOT/'scripts/query_attention_mlp_front.rs',ROOT/'scripts/wasm_patch_unique.rs',ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique',ROOT/'artifacts/wasm-audit-target/patch-unique-build.json']+list((D/'runtime').glob('*.rs'))+list(D.glob('*.rs'))+[ROOT/'scripts/prefix_state_cache.rs',ROOT/'scripts/prefix_state_cache_canister.rs',Path(__file__),ROOT/'artifacts/delta-register-v3/build/kernel.wat']
 refs={str(p.relative_to(ROOT)):sha(p) for p in sources}
 env=dict(os.environ,CARGO_MANIFEST_DIR=str(D),CARGO_PKG_NAME='imajev-runtime',CARGO_PKG_VERSION='0.1.0')
 with (D/'runtime-compiler.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 wrapper=json.loads((baseline/'report.json').read_text())['command'][:];wrapper[wrapper.index('--edition=2021')+1]=str(D/'lib.rs');wrapper[wrapper.index('-o')+1]=str(D/'raw.wasm')
 for i,arg in enumerate(wrapper):
  if arg.startswith('imajev_runtime='):wrapper[i]='imajev_runtime='+str(D/'libimajev_runtime_register.rlib')
 env.update(CARGO_PKG_NAME='imajev-inference',CARGO_PKG_VERSION_MAJOR='0',CARGO_PKG_VERSION_MINOR='1',CARGO_PKG_VERSION_PATCH='0',CARGO_PKG_VERSION_PRE='',CARGO_CRATE_NAME='imajev_inference')
 with (D/'compiler.log').open('w') as log:subprocess.run(wrapper,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 patcher=ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique';patches=[];previous=D/'raw.wasm'
 for index in range(6):
  if index<4:
   old=json.loads((baseline/f'patch{index}.json').read_text());symbol=old['export']
   wat=ROOT/'artifacts/single-quad/build-v2'/f'kernel{index}.wat'
   if index==3:wat=ROOT/'artifacts/s1-pair-bounds-v1/build/full-kernel.wat'
  elif index==4:symbol='__imajev_delta_register';wat=ROOT/'artifacts/delta-register-v3/build/kernel.wat'
  else:symbol='__imajev_f32_wide';wat=ROOT/'artifacts/f32-output64-v1/build-v2/wide.wat'
  out=D/('full.wasm' if index==5 else f'patched{index}.wasm')
  assert wat.exists(),wat
  patches.append(json.loads(subprocess.check_output([str(patcher),str(previous),str(wat),str(out),symbol],text=True)));assert len({p['function_index'] for p in patches})==len(patches),'Merged patch target';previous=out
 assert refs=={p:sha(ROOT/p) for p in refs};assert hashes=={p:sha(ROOT/p) for p in hashes}
 report=dict(baseline=provenance['module_sha256'],wasm_sha256=sha(D/'full.wasm'),source_hashes=refs,dependency_hashes=hashes,runtime_command=cmd,canister_command=wrapper,runtime_feature_set=json.loads(runtime_fp['features']),patches=patches,scope=__doc__)
 (D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(wasm_sha256=report['wasm_sha256'],bytes=(D/'full.wasm').stat().st_size)))
if __name__=='__main__':main()
