#!/usr/bin/env python3
"""Compare the actual Rust row planner to independent bit-key mapping on real inputs."""
from pathlib import Path
import hashlib,json,struct,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/row-plan-real-v1';d.mkdir(exist_ok=False);reference=ROOT/'artifacts/row-reuse-native-v1/report.json';r=json.loads(reference.read_text());assert r['complete'] and all(sha(ROOT/p)==h for p,h in r['source_hashes'].items())
 source=d/'main.rs';helper=ROOT/'scripts/projection_row_reuse.rs';source.write_text('#[path="'+str(helper)+'"]mod projection_row_reuse;\n'+r'''
use std::fs;
fn main(){let b=fs::read(std::env::args().nth(1).unwrap()).unwrap();let n=u32::from_le_bytes(b[..4].try_into().unwrap())as usize;assert!((1..=89).contains(&n));let mut at=4;
let q:Vec<i16>=b[at..at+2*n*2560].chunks_exact(2).map(|x|i16::from_le_bytes(x.try_into().unwrap())).collect();at+=2*n*2560;
let sx:Vec<f32>=b[at..at+4*n*10].chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();at+=4*n*10;
let qa:Vec<f32>=b[at..at+4*n*64].chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();at+=4*n*64;
let za:Vec<f32>=b[at..at+4*n*64].chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();at+=4*n*64;assert_eq!(at,b.len());
let p=projection_row_reuse::RowPlan::new(&q,&sx,&qa,&za,n,2560,64).unwrap();
println!("{{\"first\":{:?},\"index\":{:?}}}",p.first,p.index);
}
''');binary=d/'plan';command=['rustc','--edition=2021',str(source),'-o',str(binary)]
 compiled=subprocess.run(command,capture_output=True,text=True,check=True);(d/'compiler.log').write_text(compiled.stdout+compiled.stderr);cases=[];files=[Path(__file__),reference,source,binary,helper,d/'compiler.log']
 for c in r['cases']:
  p=ROOT/c['output'];assert sha(p)==c['output_sha256']
  with np.load(p,allow_pickle=False)as z:
   raw=struct.pack('<I',c['tokens'])+z['q'].astype('<i2').tobytes()+z['scales'].astype('<f4').tobytes()+z['qa'].astype('<f4').tobytes()+z['za'].astype('<f4').tobytes();expected=dict(first=z['first'].tolist(),index=z['index'].tolist())
  ip=d/f"{c['case']}.bin";ip.write_bytes(raw);result=subprocess.check_output([str(binary),str(ip)],text=True);op=d/f"{c['case']}.json";op.write_text(result);assert json.loads(result)==expected
  cases.append(dict(case=c['case'],tokens=c['tokens'],unique_rows=c['unique_rows'],rust_plan_equal=True,input_sha256=sha(ip),output_sha256=sha(op)));files.extend([ip,op,p])
 summary=dict(complete=True,cases=cases,command=command,scope='Real q/scale/QA/ZA from independent host embedding/norm and projections. Actual Rust planner matches Python byte-key map. No Wasm performance or full inference claim.',source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files});(d/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
 with zipfile.ZipFile(d/'frozen-tests.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(cases,indent=2))
if __name__=='__main__':main()
