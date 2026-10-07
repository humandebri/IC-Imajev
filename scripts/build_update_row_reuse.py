#!/usr/bin/env python3
"""Build first-layer exact projection sharing; all contextual operations retain all tokens."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-finite-max-v1';d=ROOT/'artifacts/update-row-reuse-v1';d.mkdir(exist_ok=False)
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/projection_row_reuse.rs';(d/helper.name).write_bytes(helper.read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-finite-max-v1','artifacts/update-row-reuse-v1')
 patch='''    p=D/'runtime/delta_head_continue.rs';text=p.read_text()
    text=text.replace('const C: usize = 2560;', 'mod projection_row_reuse;\\nuse projection_row_reuse::RowPlan;\\nconst C: usize = 2560;')
    text=text.replace('    pub(crate) gates: Vec<f32>,','    pub(crate) gates: Vec<f32>,\\n    reuse:Option<ReusedRows>,')
    before='        Ok((Self { q, qa, za, gates }, bytes))'
    assert text.count(before)==1
    text=text.replace(before,'        let reuse=if root=="model.language_model.layers.0.linear_attn" {ReusedRows::new(&q,&qa,&za)?}else{None};\\n        Ok((Self { q, qa, za, gates,reuse }, bytes))')
    before='            gates: rest[sx + 2 * a..].to_vec(),';assert text.count(before)==1
    text=text.replace(before,before+'\\n            reuse:None,')
    before='crate::evaluate_integer_with_ax(&qp, &[], m, read, Some(&prep.q), Some(&prep.qa))?';assert text.count(before)==1
    text=text.replace(before,'project_rows(&qp,prep,false,m,read)?')
    before='crate::evaluate_integer_with_ax(&zp, &[], m, read, Some(&prep.q), Some(&prep.za))?';assert text.count(before)==1
    text=text.replace(before,'project_rows(&zp,prep,true,m,read)?')
    extra=r"""
struct ReusedRows{plan:RowPlan,q:QuantizedRows,qa:Vec<f32>,za:Vec<f32>}
impl ReusedRows{
 fn new(q:&QuantizedRows,qa:&[f32],za:&[f32])->Result<Option<Self>>{
  let Some(plan)=RowPlan::new(q.values(),q.scales(),qa,za,q.rows(),C,R)else{return Ok(None)};
  let mut values=Vec::with_capacity(plan.first.len()*C);let mut scales=Vec::with_capacity(plan.first.len()*10);
  for &t in &plan.first{values.extend(q.values()[t*C..(t+1)*C].iter().map(|&v|(v as i8)as u8));scales.extend_from_slice(&q.scales()[t*10..(t+1)*10]);}
  let unique=QuantizedRows::from_bytes(plan.first.len(),C,&values,&scales)?;
  let qa=plan.gather(qa,R);let za=plan.gather(za,R);Ok(Some(Self{plan,q:unique,qa,za}))
 }
}
fn project_rows<F,B>(r:&Request,prep:&Preparation,z:bool,m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer{
 if let Some(reuse)=&prep.reuse{
  let mut unique=r.clone();unique.dims[0]=reuse.q.rows();
  let(y,bytes)=crate::evaluate_integer_with_ax(&unique,&[],m,read,Some(&reuse.q),Some(if z{&reuse.za}else{&reuse.qa}))?;
  Ok((reuse.plan.expand(&y,r.dims[1]),bytes))
 }else{crate::evaluate_integer_with_ax(r,&[],m,read,Some(&prep.q),Some(if z{&prep.za}else{&prep.qa}))}
}
"""
    p.write_text(text+extra)
    (D/'runtime/delta_head_continue').mkdir()
    (D/'runtime/delta_head_continue/projection_row_reuse.rs').write_bytes((D.parent/'projection_row_reuse.rs').read_bytes())
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 s=s.replace("sources = [D/'finite-sites.json',","sources = [D/'runtime/delta_head_continue/projection_row_reuse.rs',D/'finite-sites.json',")
 (d/'frozen-builder.py').write_text(s);files=[Path(__file__),helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=[p.name for p in (old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes()];assert changed==['delta_head_continue.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['delta_head_continue/projection_row_reuse.rs'],arithmetic_kernels_equal=True,scope='Only first-layer capture shares identical Q lanes/scales/QA/ZA rows; scatter before conv, recurrence, state and gating. Restore has no reuse and wire format unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
