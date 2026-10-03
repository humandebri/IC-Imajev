//! Immutable INT8 K4/output-pair weights. Original byte reads remain exact.
use crate::{Result,int8_kernel::QuantizedRows};
use std::{rc::Rc,cell::OnceCell};
#[cfg(feature="experimental-strassen-raw")]
#[path="strassen_raw.rs"]
pub(crate) mod strassen_raw;
#[derive(Clone)]
pub struct PreparedPairs {data:Rc<[u8]>,scales:crate::prepared_weights::PreparedF32,rows:usize,cols:usize,quad:bool}
#[derive(Clone)]
pub struct PackedView {fixed:PreparedPairs,start:usize,rows:usize}
pub struct OriginalView {fixed:PreparedPairs,start:usize,len:usize,decoded:OnceCell<Vec<u8>>}
impl PreparedPairs {
    pub fn from_le_bytes(bytes:&[u8],rows:usize,cols:usize)->Result<Self> {
        let count=rows.checked_mul(cols).ok_or("paired weight overflow")?;
        let total=count.checked_add(rows.checked_mul(4).ok_or("paired scale overflow")?).ok_or("paired byte overflow")?;
        if rows==0 || rows%32!=0 || cols==0 || cols%256!=0 || count>30_000_000 || total!=bytes.len() {return Err("paired weight shape".into());}
        let scales=crate::prepared_weights::PreparedF32::from_le_bytes(&bytes[count..])?;
        if scales.iter().any(|v|*v<=0.) {return Err("paired scale".into());}
        #[cfg(feature="experimental-strassen-raw")]
        if rows>=512 {let mut data=strassen_raw::pack(bytes,rows,cols);data.truncate(count);return Ok(Self{data:data.into(),scales,rows,cols,quad:true});}
        let mut data=Vec::with_capacity(count);
        for r in (0..rows).step_by(2) {for c in (0..cols).step_by(4) {
            data.extend_from_slice(&bytes[r*cols+c..r*cols+c+4]);
            data.extend_from_slice(&bytes[(r+1)*cols+c..(r+1)*cols+c+4]);
        }}
        Ok(Self{data:data.into(),scales,rows,cols,quad:false})
    }
    pub fn bytes(&self)->usize {self.data.len()+self.scales.len()*4}
    fn scale_bytes(&self)->&[u8] {
        #[cfg(target_endian="big")] compile_error!("Prepared scale byte views require little endian");
        // Initialized immutable F32 values; bytes are valid at alignment one.
        unsafe {std::slice::from_raw_parts(self.scales.as_ptr().cast(),self.scales.len()*4)}
    }
    pub fn weight_bytes(&self)->usize {self.rows*self.cols}
    pub fn original_view(&self,start:usize,len:usize)->Option<OriginalView> {
        if start.checked_add(len)? > self.bytes() {return None;}
        Some(OriginalView{fixed:self.clone(),start,len,decoded:OnceCell::new()})
    }
    pub fn packed_rows(&self,start:usize,rows:usize)->Option<PackedView> {
        if start%2!=0 || rows==0 || rows%8!=0 || start.checked_add(rows)?>self.rows {return None;}
        Some(PackedView{fixed:self.clone(),start:start*self.cols,rows})
    }
}
impl OriginalView {
    pub fn packed(&self)->Option<PackedView> {
        if self.start%self.fixed.cols!=0 || self.len%self.fixed.cols!=0 || self.start.checked_add(self.len)?>self.fixed.weight_bytes(){return None;}
        self.fixed.packed_rows(self.start/self.fixed.cols,self.len/self.fixed.cols)
    }
    pub fn prepared_scales(&self)->Option<crate::prepared_weights::PreparedF32> {
        let start=self.start.checked_sub(self.fixed.weight_bytes())?;
        if start%4!=0 || self.len%4!=0 {return None;}
        self.fixed.scales.slice(start/4,self.len/4)
    }
    pub fn original_was_decoded(&self)->bool {self.decoded.get().is_some()}
}
impl AsRef<[u8]> for OriginalView {
    fn as_ref(&self)->&[u8] {
        // Scale-only slices are already contiguous in the original byte order.
        if self.start>=self.fixed.weight_bytes(){let start=self.start-self.fixed.weight_bytes();return &self.fixed.scale_bytes()[start..start+self.len];}
        self.decoded.get_or_init(||(self.start..self.start+self.len).map(|i|{
            if i>=self.fixed.weight_bytes(){return self.fixed.scale_bytes()[i-self.fixed.weight_bytes()];}
            let row=i/self.fixed.cols;let c=i%self.fixed.cols;
            #[cfg(feature="experimental-strassen-raw")]
            if self.fixed.quad {return self.fixed.data[strassen_raw::index(self.fixed.rows,self.fixed.cols,row,c)];}
            self.fixed.data[(row/2)*self.fixed.cols*2+(c/4)*8+(row%2)*4+c%4]
        }).collect())
    }
}
impl crate::prepared_weights::WeightBuffer for OriginalView {
    fn prepared_f32(&self)->Option<crate::prepared_weights::PreparedF32> {self.prepared_scales()}
    fn prepared_output_pairs(&self)->Option<PackedView> {self.packed()}
}
impl PackedView {
    pub fn cols(&self)->usize {self.fixed.cols}
    pub fn rows(&self)->usize {self.rows}
    pub(crate) fn scales(&self)->&[f32] {let start=self.start/self.cols();&self.fixed.scales[start..start+self.rows]}
    fn values(&self)->&[u8] {&self.fixed.data[self.start..self.start+self.rows*self.cols()]}
}
pub(crate) fn duplicate(input:&[i16])->Vec<i16> {
    debug_assert_eq!(input.len()%4,0);
    let mut data:Vec<i16>=Vec::with_capacity(input.len()*2);
    #[cfg(target_arch="wasm32")]
    // The opaque q constructor supplies complete block256 rows. SIMD writes
    // every element of spare capacity before exposing the new length.
    unsafe {duplicate_simd(input,data.as_mut_ptr());data.set_len(input.len()*2);}
    #[cfg(not(target_arch="wasm32"))]
    for chunk in input.chunks_exact(4){data.extend_from_slice(chunk);data.extend_from_slice(chunk);}
    data
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn duplicate_simd(input:&[i16],output:*mut i16) {
    use core::arch::wasm32::*;
    for i in (0..input.len()).step_by(4){v128_store(output.add(i*2).cast(),v128_load64_splat(input.as_ptr().add(i).cast()));}
}
#[cfg(all(target_arch="wasm32",not(any(feature="experimental-pair-shared-body",feature="experimental-pair-wat"))))]
#[path="output_pairs_simd.rs"]
mod simd;
#[cfg(all(target_arch="wasm32",feature="experimental-pair-shared-body",not(feature="experimental-pair-wat")))]
#[path="output_pairs_shared.rs"]
mod simd;
#[cfg(all(target_arch="wasm32",feature="experimental-pair-wat"))]
#[path="output_pairs_wat.rs"]
mod simd;
pub fn project(q:&QuantizedRows,w:&PackedView,scales:&[f32])->Result<Vec<f32>> {
    if !scales.iter().all(|v|v.is_finite()&&*v>0.) {return Err("paired projection shape/scales".into());}
    project_checked(q,w,scales)
}
/// Both rows and positive finite scales belong to the same immutable prepared
/// tensor. Clients cannot create or mutate PackedView; no scale read/scan needed.
pub(crate) fn project_prepared(q:&QuantizedRows,w:&PackedView)->Result<Vec<f32>> {
    project_checked(q,w,w.scales())
}
fn project_checked(q:&QuantizedRows,w:&PackedView,scales:&[f32])->Result<Vec<f32>> {
    if q.cols()!=w.cols() || q.rows()==0 || q.rows()>132 || q.rows().checked_mul(w.rows()).is_none_or(|v|v>crate::MAX_FLOATS) || scales.len()!=w.rows() {return Err("paired projection shape/scales".into());}
    #[cfg(feature="experimental-strassen-raw")]
    if w.fixed.quad {return strassen_raw::project(q,w,scales);}
    #[cfg(all(target_arch="wasm32",feature="experimental-pair-direct-input"))]
    // The matching WAT ABI reads four original I16 values with load64_splat
    // at first use. No query-local duplicated buffer is constructed.
    let data=q.values();
    #[cfg(not(all(target_arch="wasm32",feature="experimental-pair-direct-input")))]
    let data=q.output_pair_values();
    let cols=w.cols();let rows=w.rows();let mut out=vec![0.;q.rows()*rows];
    for r in (0..rows).step_by(32) {
        let width=(rows-r).min(32);
        // Client query boundaries are multiples of eight, not always 32. Only
        // the final partial column tile needs scratch; never reconstruct the
        // original rows for the whole projection. No read crosses its view.
        let padded_weights;
        let padded_scales;
        let (weights,sw)=if width==32 {
            (&w.values()[r*cols..(r+32)*cols],&scales[r..r+32])
        }else {
            padded_weights={let mut v=w.values()[r*cols..].to_vec();v.resize(32*cols,0);v};
            padded_scales={let mut v=[1f32;32];v[..width].copy_from_slice(&scales[r..]);v};
            (padded_weights.as_slice(),padded_scales.as_slice())
        };
        #[cfg(all(target_arch="wasm32",feature="experimental-pair-wat"))]
        {
            // All real rows share the immutable weight expansion for each K
            // block. Each row keeps exactly the original block/F32 order.
            // Up to 132*32 F32 scratch values; no padding tokens or heap cache.
            let mut sums=vec![[0f32;32];q.rows()];
            for block in 0..cols/256 {
                unsafe {simd::accumulate(data.as_ptr(),weights.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,sw.as_ptr(),sums.as_mut_ptr().cast(),q.rows());}
            }
            for i in 0..q.rows() {out[i*rows+r..i*rows+r+width].copy_from_slice(&sums[i][..width]);}
        }
        #[cfg(not(all(target_arch="wasm32",feature="experimental-pair-wat")))]
        {
        let mut t=0;
        macro_rules! tile {($n:literal)=>{{
            let mut sums=[[0f32;32];$n];
            for block in 0..cols/256 {
                #[cfg(all(target_arch="wasm32",not(any(feature="experimental-pair-shared-body",feature="experimental-pair-wat"))))]
                // Constructors establish packed weight spans and real token
                // rows. The dispatch never computes a padding token.
                unsafe {simd::accumulate::<$n>(data.as_ptr().add(t*cols*2),weights.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(t*(cols/256)+block),cols/256,sw.as_ptr(),&mut sums);}
                #[cfg(all(target_arch="wasm32",feature="experimental-pair-shared-body",not(feature="experimental-pair-wat")))]
                unsafe {simd::accumulate(data.as_ptr().add(t*cols*2),weights.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(t*(cols/256)+block),cols/256,sw.as_ptr(),&mut sums);}
                #[cfg(all(target_arch="wasm32",feature="experimental-pair-wat"))]
                unsafe {simd::accumulate(data.as_ptr().add(t*cols*2),weights.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(t*(cols/256)+block),cols/256,sw.as_ptr(),sums.as_mut_ptr().cast(),$n);}
                #[cfg(not(target_arch="wasm32"))]
                for i in 0..$n {for j in 0..32 {let mut dot=0i32;
                    for c in block*256..(block+1)*256 {let wi=(j/2)*cols*2+(c/4)*8+(j%2)*4+c%4;dot+=data[(t+i)*cols*2+(c/4)*8+c%4]as i32*(weights[wi]as i8)as i32;}
                    sums[i][j]+=(dot as f32*q.scales()[(t+i)*(cols/256)+block])*sw[j];
                }}
            }
            for i in 0..$n {out[(t+i)*rows+r..(t+i)*rows+r+width].copy_from_slice(&sums[i][..width]);}
            t+=$n;
        }}}
        match q.rows() {
            45=>tile!(45),80=>{tile!(40);tile!(40);},87=>{tile!(44);tile!(43);},89=>{tile!(45);tile!(44);},132=>{tile!(44);tile!(44);tile!(44);},
            _=>{
                while t+48<=q.rows(){tile!(48);}
                if t+32<=q.rows(){tile!(32);}if t+16<=q.rows(){tile!(16);}if t+8<=q.rows(){tile!(8);}if t+4<=q.rows(){tile!(4);}if t+2<=q.rows(){tile!(2);}if t<q.rows(){tile!(1);}
            }
        }
        debug_assert_eq!(t,q.rows());
        }
    }
    if out.iter().any(|v|!v.is_finite()){return Err("paired projection output".into());}Ok(out)
}
#[cfg(test)]mod tests {use super::*;
    struct DirectReader(OriginalView);
    impl AsRef<[u8]> for DirectReader {fn as_ref(&self)->&[u8] {assert!(self.0.packed().is_none(),"packed integer path must avoid original decoding");assert!(self.0.prepared_scales().is_none(),"prepared scales must avoid byte decoding");self.0.as_ref()}}
    impl crate::prepared_weights::WeightBuffer for DirectReader {fn prepared_f32(&self)->Option<crate::prepared_weights::PreparedF32>{self.0.prepared_scales()}fn prepared_output_pairs(&self)->Option<PackedView>{self.0.packed()}}
    #[test]fn prepared_scale_views_preserve_capacity_bits_and_lifetime() {
        let (rows,cols)=(512,256);let mut bytes=vec![0u8;rows*cols];
        let values:Vec<f32>=(0..rows).map(|i|if i%2==0 {f32::MIN_POSITIVE}else{0.00123*(i+1)as f32}).collect();
        for v in &values {bytes.extend(v.to_le_bytes());}
        let fixed=PreparedPairs::from_le_bytes(&bytes,rows,cols).unwrap();
        assert_eq!(fixed.bytes(),bytes.len());assert_eq!(fixed.data.len(),rows*cols);
        let view=fixed.original_view(rows*cols+31*4,33*4).unwrap();let scales=view.prepared_scales().unwrap();
        assert!(core::ptr::eq(scales.as_ptr(),fixed.scales[31..].as_ptr()));
        assert_eq!(scales.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),values[31..64].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(!view.original_was_decoded());
        assert!(fixed.original_view(rows*cols-1,4).unwrap().prepared_scales().is_none());
        assert!(fixed.original_view(rows*cols+1,4).unwrap().prepared_scales().is_none());
        assert!(fixed.original_view(rows*cols,3).unwrap().prepared_scales().is_none());
        drop(fixed);assert_eq!(view.as_ref(),&bytes[rows*cols+31*4..rows*cols+64*4]);
        assert_eq!(scales.len(),33);
        for value in [0.,-0.,-1.,f32::NAN,f32::INFINITY] {
            let mut invalid=bytes.clone();invalid[rows*cols..rows*cols+4].copy_from_slice(&value.to_le_bytes());
            assert!(PreparedPairs::from_le_bytes(&invalid,rows,cols).is_err());
        }
    }
    #[test]fn reader_dispatch_and_nonzero_row_ranges() {
        let rows=64;let cols=256;let mut bytes:Vec<u8>=(0..rows*cols).map(|i|(i%251)as u8).collect();for i in 0..rows {bytes.extend((0.00123*(i+1)as f32).to_le_bytes());}
        let w=PreparedPairs::from_le_bytes(&bytes,rows,cols).unwrap();
        let m=crate::Manifest{version:1,model:"a".repeat(64),pack_hash:"b".repeat(64),bytes:bytes.len()as u64,tensors:vec![crate::Tensor{name:"base".into(),offset:0,rows,cols,dtype:"int8".into(),bytes:bytes.len()as u64}]};
        for (n,start,tile) in [(1,0,32),(87,32,32),(132,0,64),(132,8,40),(45,8,56),(87,24,8),(7,1,8)] {
            let r:crate::Request=serde_json::from_value(serde_json::json!({"version":1,"model":m.model,"pack_hash":m.pack_hash,"input_hash":"c".repeat(64),"step":0,"op":"int8_matmul","tensor":"base","dims":[n,tile,cols,start],"scalars":[]})).unwrap();
            let x:Vec<f32>=(0..n*cols).map(|i|(i%79)as f32-39.).collect();
            let old=crate::evaluate_with_buffer(&r,&x,&m,|o,l|Ok(bytes[o as usize..o as usize+l].to_vec())).unwrap();
            let new=crate::evaluate_with_prepared_buffer(&r,&x,&m,|o,l|Ok(DirectReader(w.original_view(o as usize,l).unwrap()))).unwrap();
            assert_eq!(old.1,new.1);assert!(old.0.iter().zip(new.0).all(|(a,b)|a.to_bits()==b.to_bits()));
        }
    }
    #[test]fn original_byte_slices_and_lifetime(){let rows=64;let cols=256;let mut bytes:Vec<u8>=(0..rows*cols).map(|i|(i%256)as u8).collect();for _ in 0..rows{bytes.extend(0.123f32.to_le_bytes());}let w=PreparedPairs::from_le_bytes(&bytes,rows,cols).unwrap();assert_eq!(w.bytes(),bytes.len());let view=w.original_view(1,bytes.len()-1).unwrap();assert!(view.packed().is_none());for (s,n) in [(0,32*cols),(31*cols,cols+3),(rows*cols-3,64*4+3),(rows*cols,64*4)] {let v=w.original_view(s,n).unwrap();assert_eq!(v.as_ref(),&bytes[s..s+n]);}drop(w);assert_eq!(view.as_ref(),&bytes[1..]);}
    #[test]fn real_rows_scales_extremes(){for n in [1,7,8,32,45,48,64,80,81,87,88,89,132] {let cols=512;let rows=32;let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]*(1+i/256%3)as f32).collect();let raw:Vec<i8>=(0..rows*cols).map(|i|[-128,127,1,-1,0][i%5]).collect();let sw:Vec<f32>=(0..rows).map(|i|0.0123*(i+1)as f32).collect();let mut bytes:Vec<u8>=raw.iter().map(|v|*v as u8).collect();for v in &sw {bytes.extend(v.to_le_bytes());}let w=PreparedPairs::from_le_bytes(&bytes,rows,cols).unwrap();let v=w.original_view(0,rows*cols).unwrap();let q=crate::int8_kernel::quantize_rows(&x,n,cols).unwrap();let old=crate::int8_kernel::project(&q,&raw,&sw,rows).unwrap();let new=project(&q,&v.packed().unwrap(),&sw).unwrap();assert!(old.iter().zip(new).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(!v.original_was_decoded());assert_eq!(q.output_pair_values().len(),n*cols*2);}}
 #[cfg(feature="experimental-strassen-raw")]
 #[test]fn raw_quadrants_preserve_original_ranges_scales_and_shifted_row_views(){
  let(rows,cols)=(512,256);let raw:Vec<i8>=(0..rows*cols).map(|i|(i*17+i/cols*31)as u8 as i8).collect();let scales:Vec<f32>=(0..rows).map(|i|0.00123*(i+1)as f32).collect();let mut bytes:Vec<u8>=raw.iter().map(|&v|v as u8).collect();for v in &scales{bytes.extend(v.to_le_bytes());}let w=PreparedPairs::from_le_bytes(&bytes,rows,cols).unwrap();assert!(w.quad);assert_eq!(w.bytes(),bytes.len());for(s,len)in[(0,bytes.len()),(1,65535),(rows*cols-3,67)]{assert_eq!(w.original_view(s,len).unwrap().as_ref(),&bytes[s..s+len]);}
  for(n,start,tile)in[(1,0,32),(7,2,8),(87,2,32),(89,24,8),(132,480,32),(45,502,8)]{
   let x:Vec<f32>=(0..n*cols).map(|i|[-127.,127.,0.,-0.,1.,-1.][i%6]).collect();let q=crate::int8_kernel::quantize_rows(&x,n,cols).unwrap();let v=w.original_view(start*cols,tile*cols).unwrap();let new=project(&q,&v.packed().unwrap(),&scales[start..start+tile]).unwrap();let old=crate::int8_kernel::project(&q,&raw[start*cols..(start+tile)*cols],&scales[start..start+tile],tile).unwrap();assert!(new.iter().zip(old).all(|(a,b)|a.to_bits()==b.to_bits()),"n={n} start={start}");assert!(!v.original_was_decoded());let a=q.strassen_operands();assert!(core::ptr::eq(a,q.strassen_operands()));
  }
 }

}
