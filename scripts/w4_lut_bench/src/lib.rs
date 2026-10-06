mod kernel;
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::cell::RefCell;
struct Fixed{owner:Principal,rows:usize,cols:usize,w:Vec<i8>,scales:Vec<f32>,q4:Vec<i8>,s4:Vec<f32>,packed:Vec<u8>,sealed:bool}
thread_local!{static FIXED:RefCell<Option<Fixed>>=const{RefCell::new(None)};}
#[ic_cdk::init] fn init(owner:Principal,rows:u32,cols:u32){
    assert_ne!(owner,Principal::anonymous());assert!(rows>0&&rows<=8192&&rows%16==0&&cols>0&&cols<=9216&&cols%256==0);
    FIXED.with(|s|*s.borrow_mut()=Some(Fixed{owner,rows:rows as usize,cols:cols as usize,w:vec![],scales:vec![],q4:vec![],s4:vec![],packed:vec![],sealed:false}));
}
#[derive(CandidType,Deserialize,Serialize)] pub struct Preparation{instructions:u64,bytes:u64,rows:u64}
#[ic_cdk::update] fn prepare_chunk(start:u32,bytes:Vec<u8>)->Preparation{FIXED.with(|s|{
    let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(!f.sealed&&start as usize==f.scales.len());
    assert!(!bytes.is_empty()&&bytes.len()<=1_500_000&&bytes.len()%(f.cols+4)==0);
    let count=bytes.len()/(f.cols+4);assert!(f.scales.len()+count<=f.rows);let begin=ic_cdk::api::performance_counter(0);
    f.w.extend(bytes[..count*f.cols].iter().map(|v|*v as i8));
    for b in bytes[count*f.cols..].chunks_exact(4){let v=f32::from_le_bytes(b.try_into().unwrap());assert!(v.is_finite()&&v>0.);f.scales.push(v);}
    Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:bytes.len()as u64,rows:count as u64}
})}
#[ic_cdk::update] fn seal()->Preparation{FIXED.with(|s|{
    let mut s=s.borrow_mut();let f=s.as_mut().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(!f.sealed&&f.scales.len()==f.rows);
    let begin=ic_cdk::api::performance_counter(0);let blocks=f.cols/256;
    for r in 0..f.rows {for b in 0..blocks {let weights=&f.w[r*f.cols+b*256..r*f.cols+(b+1)*256];
        let peak=weights.iter().map(|v|(*v as i16).abs()).max().unwrap();let factor=if peak==0{1.}else{peak as f32/7.};
        f.s4.push(factor*f.scales[r]);f.q4.extend(weights.iter().map(|v|(*v as f32/factor).round_ties_even().clamp(-7.,7.)as i8));
    }}f.packed=kernel::pack(&f.q4,f.rows,f.cols);f.sealed=true;
    Preparation{instructions:ic_cdk::api::performance_counter(0)-begin,bytes:(f.packed.len()+f.s4.len()*4)as u64,rows:f.rows as u64}
})}
#[derive(CandidType,Deserialize,Serialize)] pub struct Measurement{digest:Vec<u8>,quantize_instructions:u64,input_prepare_instructions:u64,project_instructions:u64,total_instructions:u64,output_values:u64,heap_pages:u64}
#[ic_cdk::query] fn project(input:Vec<u8>,method:u8)->Measurement{FIXED.with(|s|{
    let s=s.borrow();let f=s.as_ref().unwrap();assert_eq!(f.owner,ic_cdk::api::msg_caller());assert!(f.sealed&&method<16);
    // Output1024 per query; the caller carries tile progress. No query state.
    let start=(method as usize/2)*1024;assert!(start<f.rows);
    let rows=1024.min(f.rows-start);let use_lut=method%2==1;
    assert!(!input.is_empty()&&input.len()<=1_500_000&&input.len()%(f.cols*4)==0);let n=input.len()/(f.cols*4);assert!(n<=132);
    let begin=ic_cdk::api::performance_counter(0);let x:Vec<f32>=input.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
    let (q,sx)=kernel::quantize(&x,f.cols);let quant=ic_cdk::api::performance_counter(0);
    let tab=if use_lut{kernel::tables(&q)}else{vec![]};let prep=ic_cdk::api::performance_counter(0);
    let blocks=f.cols/256;let scales=&f.s4[start*blocks..(start+rows)*blocks];
    let out=if use_lut{kernel::lut(&q,&sx,&f.packed[start*f.cols/2..(start+rows)*f.cols/2],scales,n,rows,f.cols,&tab)}else{kernel::dense(&q,&sx,&f.q4[start*f.cols..(start+rows)*f.cols],scales,n,rows,f.cols)};
    let end=ic_cdk::api::performance_counter(0);let mut digest=Sha256::new();for v in &out{digest.update(v.to_le_bytes());}
    Measurement{digest:digest.finalize().to_vec(),quantize_instructions:quant-begin,input_prepare_instructions:prep-quant,project_instructions:end-prep,total_instructions:end-begin,output_values:out.len()as u64,heap_pages:core::arch::wasm32::memory_size(0)as u64}
})}
