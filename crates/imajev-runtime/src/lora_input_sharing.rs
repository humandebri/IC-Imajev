//! Query-local F32 input packing shared by distinct LoRA A matrices.
//! Products are still computed separately, in the original column order.
use crate::{Manifest, Request, Result, prepared_weights::WeightBuffer};

struct Group { start: usize, valid: usize, lanes: usize, offset: usize }
struct PackedInput { n: usize, cols: usize, values: Vec<f32>, groups: Vec<Group> }
pub(crate) fn products(x:&[f32], first:&[f32], second:&[f32], n:usize, cols:usize)
    ->Result<(Vec<f32>,Vec<f32>)> {
    let packed=PackedInput::new(x,n,cols)?;
    Ok((packed.project(first,64)?,packed.project(second,64)?))
}
impl PackedInput {
    fn new(x: &[f32], n: usize, cols: usize) -> Result<Self> {
        if n < 4 || n > 89 || cols == 0 || n.checked_mul(cols) != Some(x.len())
            || x.len() > crate::MAX_FLOATS { return Err("shared LoRA input bounds".into()); }
        let main = if n >= 32 {32} else {16};
        let mut groups = Vec::new(); let mut values = Vec::with_capacity((n+3)*cols);
        let mut start = 0;
        while start < n {
            let left = n-start;
            let lanes = if left >= main {main} else if left >= 16 {16}
                else if left >= 8 {8} else {4};
            let valid = left.min(lanes); let offset = values.len();
            for c in 0..cols {for lane in 0..lanes {
                values.push(if lane < valid {x[(start+lane)*cols+c]} else {0.});
            }}
            groups.push(Group {start, valid, lanes, offset}); start += valid;
        }
        Ok(Self {n, cols, values, groups})
    }
    #[inline(never)]
    fn project(&self, w: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows == 0 || rows.checked_mul(self.cols) != Some(w.len())
            || self.n.checked_mul(rows).is_none_or(|s| s > crate::MAX_FLOATS) {
            return Err("shared LoRA weight/output bounds".into());
        }
        let mut out = vec![0.;self.n*rows];
        for group in &self.groups {
            let packed = &self.values[group.offset..group.offset+self.cols*group.lanes];
            match group.lanes {32=>self.compute::<8>(packed,w,rows,group,&mut out),
                16=>self.compute::<4>(packed,w,rows,group,&mut out),
                8=>self.compute::<2>(packed,w,rows,group,&mut out),
                4=>self.compute::<1>(packed,w,rows,group,&mut out),_=>unreachable!()}
        }
        Ok(out)
    }
    #[inline(never)]
    fn compute<const G: usize>(&self, packed: &[f32], w: &[f32], rows: usize, group: &Group, out: &mut [f32]) {
        let mut row = 0;
        while row+16 <= rows {
            let dots = dot::<16,G>(packed,&w[row*self.cols..(row+16)*self.cols],self.cols);
            for j in 0..16 {for lane in 0..group.valid {
                out[(group.start+lane)*rows+row+j]=dots[j][lane/4][lane%4];
            }} row+=16;
        }
        while row < rows {
            let dots = dot::<1,G>(packed,&w[row*self.cols..(row+1)*self.cols],self.cols);
            for lane in 0..group.valid {out[(group.start+lane)*rows+row]=dots[0][lane/4][lane%4];} row+=1;
        }
    }
}
#[inline(never)]
fn dot<const R: usize,const G: usize>(packed: &[f32],w: &[f32],cols: usize)->[[[f32;4];G];R] {
    #[cfg(target_arch="wasm32")]
    // Private PackedInput establishes complete initialized groups; project
    // validates weights and compute supplies exactly R full rows.
    return unsafe {crate::dot_multi::<R,G>(packed,w,cols)};
    #[cfg(not(target_arch="wasm32"))] {
        let mut out=[[[0.;4];G];R];
        for c in 0..cols {for r in 0..R {for g in 0..G {for lane in 0..4 {
            out[r][g][lane]+=packed[c*G*4+g*4+lane]*w[r*cols+c];
        }}}} out
    }
}

pub(crate) fn pair_a<F,B>(x:&[f32], requests:&[Request],m:&Manifest,read:&mut F)
    ->Result<(Vec<f32>,Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
    if requests.len()!=2 {return Err("shared LoRA request pair".into());}
    let first=&requests[0]; let n=first.dims[0]; let cols=first.dims[2];
    let mut packed:Option<PackedInput>=None; let mut products=Vec::new();let mut bytes=0;
    for request in requests {
        let a=m.tensors.iter().find(|t|t.name==request.aux[0]).ok_or("shared LoRA A")?;
        if a.rows!=64 || a.cols!=cols {return Err("shared LoRA A shape".into());}
        let mut ar=request.clone();ar.op="matmul".into();ar.dims=vec![n,a.rows,cols];
        let(w,used)=crate::load_prepared_weight(a,&ar,&mut *read)?;
        let product=crate::profile::measure("lora_matmul_A_shared",|| {
            #[cfg(feature="experimental-f32-output-generic")]
            if let Some(result)=crate::matrix_prepared(x,&w,n,a.rows,cols) {return result;}
            if packed.is_none() {packed=Some(PackedInput::new(x,n,cols)?);}
            packed.as_ref().unwrap().project(&w,a.rows)
        })?;
        products.push(product);bytes+=used;
    }
    let second=products.pop().unwrap();let first=products.pop().unwrap();Ok((first,second,bytes))
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn distinct_weights_and_token_tails_preserve_scalar_bits() {
        for n in [4,7,8,16,31,32,45,64,80,87,88,89] {
            let cols=13;
            let x:Vec<_>=(0..n*cols).map(|i|if i%29==0 {-0.}else{((i*17%101)as f32-50.)/33.}).collect();
            let packed=PackedInput::new(&x,n,cols).unwrap();
            for shift in [3,11] {
                let w:Vec<_>=(0..64*cols).map(|i|((i*shift%79)as f32-39.)/31.).collect();
                let got=packed.project(&w,64).unwrap();let expected=crate::matrix_reference(&x,&w,n,64,cols).unwrap();
                assert!(got.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n}, shift={shift}");
            }
        }
    }
    #[test] fn invalid_bounds_and_weights_are_rejected() {
        for (n,cols) in [(0,13),(3,13),(513,13),(4,0),(4,usize::MAX)] {assert!(PackedInput::new(&[],n,cols).is_err());}
        let packed=PackedInput::new(&[0.;52],4,13).unwrap();assert!(packed.project(&[0.;12],1).is_err());
        assert!(packed.project(&[],0).is_err());assert!(packed.project(&[],usize::MAX).is_err());
    }
}
