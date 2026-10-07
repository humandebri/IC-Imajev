pub fn gated_norm_scalar(values:Vec<f32>,gate:Vec<f32>)->Vec<f32>{
    values.into_iter().zip(gate).map(|(v,g)|bf(bf(v)*silu(g))).collect()
}
pub fn gated_norm_owned(mut values:Vec<f32>,gate:Vec<f32>)->Vec<f32>{
    assert_eq!(values.len(),gate.len());
    #[cfg(target_arch="wasm32")]
    {prepared_activation::with_table::<3,_>(|table|unsafe{gated_norm_simd(&mut values,&gate,table)});return values;}
    #[cfg(not(target_arch="wasm32"))]
    gated_norm_scalar(values,gate)
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn gated_norm_simd(values:&mut[f32],gate:&[f32],table:Option<&[f32;65536]>){
    use core::arch::wasm32::*;
    #[inline(always)]unsafe fn round(v:v128)->v128{let parity=v128_and(u32x4_shr(v,16),i32x4_splat(1));v128_and(i32x4_add(v,i32x4_add(i32x4_splat(0x7fff),parity)),i32x4_splat(0xffff0000u32 as i32))}
    let emit=|g:f32|{let bits=g.to_bits();if bits&65535==0 && g.is_finite(){if let Some(t)=table{return t[(bits>>16)as usize];}}silu_original(g)};
    let mut i=0;
    while i+4<=values.len(){let p=values.as_mut_ptr().add(i);let g=gate.as_ptr().add(i);let v=round(v128_load(p.cast()));let s=f32x4(emit(*g),emit(*g.add(1)),emit(*g.add(2)),emit(*g.add(3)));v128_store(p.cast(),round(f32x4_mul(v,s)));i+=4;}
    while i<values.len(){values[i]=bf(bf(values[i])*emit(gate[i]));i+=1;}
}
