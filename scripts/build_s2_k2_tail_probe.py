#!/usr/bin/env python3
"""Benchmark rank49 compact K2 direct outputs against actual current full K2 runtime."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/s2-k2-tail-probe-v1';d.mkdir(exist_ok=False);b=d/'build';src=b/'src';src.mkdir(parents=True)
 kernels=ROOT/'artifacts/s2-k2-tail-kernels-v1';audit=json.loads((kernels/'layout-audit.json').read_text());assert audit['complete']and audit['all_integer_roots_equal']
 kr=json.loads((kernels/'report.json').read_text());normal=ROOT/'artifacts/update-k2-compact-v1';nb=json.loads((normal/'build/report.json').read_text())
 for key in ['source_hashes','dependency_hashes']:assert all(sha(ROOT/p)==h for p,h in nb[key].items())
 assert all(sha(ROOT/p)==h for p,h in kr['source_hashes'].items())
 ns=dict(__name__='rank49',__file__=str(kernels/'plan.py'));exec(compile((kernels/'plan.py').read_text(),str(kernels/'plan.py'),'exec'),ns)
 ad,bd,cd,leaves,roots,_=ns['plan']()
 coeff=''
 for name,dag,names,kind in [('A',ad,[a for a,_ in leaves],'i16'),('B',bd,[b for _,b in leaves],'i16'),('C',cd,[v for row in roots for v in row],'i32')]:
  coeff+=f'pub(super)const {name}:[&[(usize,{kind})];{len(names)}]=[\n'+''.join('&['+','.join(f'({i},{v})'for i,v in sorted(dag.symbols[n].items()))+'],\n'for n in names)+'];\n'
 (src/'coeff.rs').write_text(coeff)
 prep='use core::arch::wasm32::*;\n#[target_feature(enable="simd128")]pub(super)unsafe fn inputs(q:&imajev_runtime::int8_kernel::QuantizedRows,groups:usize,out:*mut i16){let cols=q.cols();for group in 0..groups{for block in 0..cols/256{for k in(0..64).step_by(8){\n'
 for i in range(16):prep+=f'let x{i}=v128_load(q.values().as_ptr().add((group*4+{i//4})*cols+block*256+{i%4*64}+k).cast());\n'
 def name(n):return 'x'+n[1:]if n.startswith('a')and not n.startswith('ad')else n
 for n,terms in ad.nodes:
  entries=list(terms.items());p,v=entries[0];value=name(p)if v>0 else 'i16x8_sub(i16x8_splat(0),'+name(p)+')'
  for p,v in entries[1:]:value=f'i16x8_{"add"if v>0 else "sub"}({value},{name(p)})'
  prep+=f'let {n}={value};\n'
 for m,(n,_)in enumerate(leaves):prep+=f'v128_store(out.add(({m}*groups+group)*(cols/4)+block*64+k).cast(),{name(n)});\n'
 prep+='}}}}\n';(src/'prepare.rs').write_text(prep)
 base=(ROOT/'scripts/wat_s2_lane_bench/src/s2.rs').read_text();a=base.index(' pub fn project(');z=base.index('\npub fn operands',a)
 project=''' pub fn project_wide(&self,q:&QuantizedRows,a:&Operands,sw:&[f32],rows:usize)->Result<Vec<f32>>{
  if q.rows()==0||q.rows()>132||q.rows()!=a.rows||q.cols()!=a.cols||q.cols()!=self.cols||rows==0||rows%32!=0||rows>self.rows||q.rows()*rows>900_000||sw.len()!=rows||!sw.iter().all(|v|v.is_finite()&&*v>0.){return Err("K2 rank49 shape".into());}
  let cols=self.cols;let blocks=cols/256;
  #[cfg(target_arch="wasm32")]
  let out=unsafe{
   let len=q.rows()*rows;let mut out=Vec::<core::mem::MaybeUninit<f32>>::with_capacity(len);out.set_len(len);
   let ap:[*const i16;49]=core::array::from_fn(|m|a.data.as_ptr().add(m*a.groups*(cols/4)));let plane=self.rows*cols/16;let mut r=0;
   while r<rows{let tile=if rows-r>=80{80}else{16};let wp:[*const i8;16]=core::array::from_fn(|m|self.data.as_ptr().add(m*plane+(r/16)*cols));
    for block in 0..blocks{let f=match(tile,block==0){(80,true)=>crate::s2_kernel::tile80_seed,(80,false)=>crate::s2_kernel::tile80,(16,true)=>crate::s2_kernel::tile16_seed,_=>crate::s2_kernel::tile16};f(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),rows,sw.as_ptr().add(r),out.as_mut_ptr().cast::<f32>().add(r),q.rows());}r+=tile;
   }
   let mut out=core::mem::ManuallyDrop::new(out);Vec::from_raw_parts(out.as_mut_ptr().cast::<f32>(),out.len(),out.capacity())
  };
  #[cfg(not(target_arch="wasm32"))]
  let out={let mut out=vec![0.;q.rows()*rows];for group in 0..a.groups{for row in 0..rows/4{for block in 0..blocks{
   let mut p=[0i32;49];for m in 0..49{for k in 0..64{let value:i16=coeff::B[m].iter().map(|&(j,c)|c*self.data[j*(self.rows*cols/16)+(row/4)*cols+block*256+(k/2)*8+(row%4)*2+k%2]as i16).sum();p[m]+=a.data[(m*a.groups+group)*(cols/4)+block*64+k]as i32*value as i32;}}
   for ti in 0..4{let t=group*4+ti;if t<q.rows(){for ri in 0..4{let r=row*4+ri;let dot:i32=coeff::C[ti*4+ri].iter().map(|&(m,c)|c*p[m]).sum();out[t*rows+r]+=(dot as f32*q.scales()[t*blocks+block])*sw[r];}}}
  }}}out};Ok(out)
 }
}
'''
 base=base[:a]+project+base[z:];base=base[:base.index('\n#[cfg(test)]')]
 base=base.replace('let len=49*groups*cols;','let len=49*groups*(cols/4);').replace('for lane in 0..4{data[(m*groups+group)*cols+block*256+(k/2)*8+lane*2+k%2]=value;}','data[(m*groups+group)*(cols/4)+block*64+k]=value;')
 assert base.count('for lane in 0..4{data[')==1;(src/'s2.rs').write_text(base)
 lib=(ROOT/'artifacts/s1-k2-latest-control-v1/build/src/lib.rs').read_text().replace('winograd','s2').replace('mod win_kernel','mod s2_kernel')
 lib=lib.replace('#[cfg(target_arch="wasm32")]\nmod kernel;','')
 lib=lib.replace('let wa=(method==4).then(||s2::operands(&q));','let wa=(method==4&&n>1).then(||s2::operands(&q));').replace('let out=if method==4{','let out=if method==4&&n>1{')
 (src/'lib.rs').write_text(lib)
 stubs=''
 for i,k in enumerate(kr['kernels']):
  tile=int(Path(k['path']).stem.split('_')[0]);seed='_seed'if '_seed'in k['symbol']else '';stubs+=f'#[export_name="{k["symbol"]}"]#[inline(never)]pub(crate)unsafe extern "C" fn tile{tile}{seed}(q:usize,w:usize,cols:usize,start:usize,sx:usize,stride:usize,sw:usize,out:*mut f32,n:usize){{let marker=core::hint::black_box(q^w^cols^start^sx^stride^sw^out as usize^n)as u32;for p in 0..n*{tile}{{core::ptr::write_volatile(out.add(p),f32::from_bits((marker^{i+1})|0x7fc00000));}}}}\n'
 # Rust caller passes pointer parameters, matching the nine-i32 Wasm ABI.
 stubs=stubs.replace('q:usize,w:usize','q:*const i16,w:*const i8').replace('sx:usize','sx:*const f32').replace('sw:usize','sw:*const f32').replace('q^w^cols^start^sx^stride^sw^out as usize','q as usize^w as usize^cols^start^sx as usize^stride^sw as usize^out as usize')
 (src/'s2_kernel.rs').write_text(stubs)
 cmd=nb['command'][:];cmd[cmd.index('--edition=2021')+1]=str(src/'lib.rs');cmd[cmd.index('-o')+1]=str(b/'raw.wasm');oldenv=json.loads((ROOT/'artifacts/s1-k2-latest-control-v1/build/report.json').read_text())['explicit_env']
 with (b/'compiler.log').open('w')as f:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**oldenv),stdout=f,stderr=f,check=True)
 mapping={sha(p):p for p in list(normal.glob('*.wat'))+list((normal/'build').glob('*.wat'))+[ROOT/'artifacts/single-quad/build-v2/kernel0.wat',ROOT/'artifacts/single-quad/build-v2/kernel2.wat']}
 paths=[(mapping[p['source_sha256']],p['export'])for p in nb['patches']]+[(ROOT/k['path'],k['symbol'])for k in kr['kernels']]
 previous=b/'raw.wasm';patches=[]
 for i,(path,symbol)in enumerate(paths):
  target=b/('diagnostic.wasm'if i==len(paths)-1 else f'patched{i}.wasm');patch=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(previous),str(path),str(target),symbol],text=True));assert patch['wasmparser_validation'];patches.append(patch);previous=target
 files=[Path(__file__),kernels/'layout-audit.json',kernels/'report.json',normal/'build/report.json']+list(src.glob('*.rs'))+[p for p,_ in paths];deps=[Path(cmd[i+1].split('=',1)[1])for i,v in enumerate(cmd)if v=='--extern']
 report=dict(wasm_sha256=sha(previous),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},dependency_hashes={str(p.relative_to(ROOT)):sha(p)for p in deps},command=cmd,explicit_env=oldenv,patches=patches,locals=max(k['locals']for k in kr['kernels']),output_tile=80,current_full_k2_control=True,single_token_current_k2_fallback=True,scope=__doc__)
 (b/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(module=report['wasm_sha256'],patches=len(patches),locals=report['locals'])))
if __name__=='__main__':main()
