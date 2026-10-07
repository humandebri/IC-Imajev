#!/usr/bin/env python3
"""Connect exact first-layer projection sharing to the actual paid Delta path."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-finite-max-v1';d=ROOT/'artifacts/update-row-reuse-v2';d.mkdir(exist_ok=False)
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/projection_row_reuse.rs';(d/helper.name).write_bytes(helper.read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-finite-max-v1','artifacts/update-row-reuse-v2')
 patch='''    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod projection_row_reuse;\\n')
    (D/'runtime/projection_row_reuse.rs').write_bytes((D.parent/'projection_row_reuse.rs').read_bytes())
    p=D/'runtime/delta_full_log.rs';text=p.read_text()
    anchor=' let(mixed,used)=crate::evaluate_integer_with_ax(&qp,original,m,read,Some(&q),Some(&qa))?;bytes+=used;'
    assert text.count(anchor)==1
    text=text.replace(anchor,' let reuse=if root=="model.language_model.layers.0.linear_attn" {ReusedRows::new(&q,&qa,&za)?}else{None};\\n let(mixed,used)=project_rows(&qp,original,&q,&qa,reuse.as_ref(),m,read)?;bytes+=used;')
    anchor=' let(z,used)=crate::evaluate_integer_with_ax(&zp,original,m,read,Some(&q),Some(&za))?;bytes+=used;'
    assert text.count(anchor)==1
    text=text.replace(anchor,' let(z,used)=project_rows(&zp,original,&q,&za,reuse.as_ref(),m,read)?;bytes+=used;')
    extra=r"""
use crate::{projection_row_reuse::RowPlan,int8_kernel::QuantizedRows};
struct ReusedRows{plan:RowPlan,q:QuantizedRows}
impl ReusedRows{
 fn new(q:&QuantizedRows,qa:&[f32],za:&[f32])->Result<Option<Self>>{
  let Some(plan)=RowPlan::new(q.values(),q.scales(),qa,za,q.rows(),COLS,64)else{return Ok(None)};
  let mut values=Vec::with_capacity(plan.first.len()*COLS);let mut scales=Vec::with_capacity(plan.first.len()*10);
  for &t in &plan.first{values.extend(q.values()[t*COLS..(t+1)*COLS].iter().map(|&v|(v as i8)as u8));scales.extend_from_slice(&q.scales()[t*10..(t+1)*10]);}
  let unique=QuantizedRows::from_bytes(plan.first.len(),COLS,&values,&scales)?;Ok(Some(Self{plan,q:unique}))
 }
}
fn project_rows<F,B>(r:&Request,original:&[f32],q:&QuantizedRows,ax:&[f32],reuse:Option<&ReusedRows>,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer{
 if let Some(reuse)=reuse{
  let mut unique=r.clone();unique.dims[0]=reuse.q.rows();let unique_ax=reuse.plan.gather(ax,64);
  let(y,bytes)=crate::evaluate_integer_with_ax(&unique,&[],m,read,Some(&reuse.q),Some(&unique_ax))?;
  Ok((reuse.plan.expand(&y,r.dims[1]),bytes))
 }else{crate::evaluate_integer_with_ax(r,original,m,read,Some(q),Some(ax))}
}
"""
    p.write_text(text+extra)
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in (old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['delta_full_log.rs','lib.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['projection_row_reuse.rs'],arithmetic_kernels_equal=True,scope='Actual paid scheduler uses bound DeltaHybrid -> delta_full_log::evaluate_from_state. First-layer QKV/Z projection sharing only; scatter before unchanged conv/gates/recurrence/state. No changes to delta_head_continue.'),indent=2)+'\n')
if __name__=='__main__':main()
