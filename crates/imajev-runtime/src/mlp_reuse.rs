//! Exact reuse of block256 input and both original F32 LoRA-A products.
use crate::{int8_kernel, prepared_weights::WeightBuffer, Manifest, Request, Result};
fn projections(r:&Request,m:&Manifest)->Result<[Request;2]> {
    if r.encoding!=crate::projection_codec::NAME || r.dims.len()!=5 || r.aux.len()!=1
        || r.scalars.len()!=1 || !r.scalars[0].is_finite()
        || !r.tensor.ends_with(".mlp.gate_proj.weight")
        || r.aux[0]!=r.tensor.replace(".gate_proj.weight",".up_proj.weight") {
        return Err("MLP reuse metadata".into());
    }
    let (n,rows,cols,start,rank)=(r.dims[0],r.dims[1],r.dims[2],r.dims[3],r.dims[4]);
    if n==0 || n>512 || rows==0 || rows%8!=0 || cols==0 || cols%256!=0
        || rank==0 || rank>256 || n.checked_mul(cols).is_none_or(|v|v>crate::MAX_FLOATS)
        || n.checked_mul(rows).is_none_or(|v|v>crate::MAX_FLOATS)
        || rows.checked_mul(cols).is_none_or(|v|v>30_000_000)
        || 2*n as u64*(rows as u64*cols as u64+rank as u64*(rows+cols) as u64)>4_500_000_000 {
        return Err("MLP reuse shape/work".into());
    }
    let requests=[r.tensor.clone(),r.aux[0].clone()].map(|name| {
        let prefix=name.strip_suffix(".weight").unwrap();
        let mut request=r.clone();request.op="lora_integer".into();request.dims.truncate(4);
        request.aux=vec![format!("{prefix}.lora_A.weight"),format!("{prefix}.lora_B.weight")];
        request.tensor=name;request
    });
    for request in &requests {
        let base=m.tensors.iter().find(|t|t.name==request.tensor).ok_or("MLP reuse base")?;
        let a=m.tensors.iter().find(|t|t.name==request.aux[0]).ok_or("MLP reuse A")?;
        let b=m.tensors.iter().find(|t|t.name==request.aux[1]).ok_or("MLP reuse B")?;
        if base.dtype!="int8" || base.cols!=cols || a.dtype!="f32" || b.dtype!="f32"
            || a.rows!=rank || a.cols!=cols || b.cols!=rank
            || start.checked_add(rows).is_none_or(|end|end>base.rows || end>b.rows) {
            return Err("MLP reuse sealed shape".into());
        }
    }
    Ok(requests)
}
fn output<F,B>(requests:&[Request;2],q:&int8_kernel::QuantizedRows,ax:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    let rank=m.tensors.iter().find(|t|t.name==requests[0].aux[0]).ok_or("MLP reuse A")?.rows;
    let n=requests[0].dims[0];let cols=requests[0].dims[2];
    if q.rows()!=n || q.cols()!=cols || ax.len()!=2*n*rank {return Err("MLP reuse prepared state shape".into());}
    let (gate,rg)=crate::evaluate_integer_with_ax(&requests[0],&[],m,read,Some(q),Some(&ax[..n*rank]))?;
    let (up,ru)=crate::evaluate_integer_with_ax(&requests[1],&[],m,read,Some(q),Some(&ax[n*rank..]))?;
    let out:Vec<_>=gate.into_iter().zip(up).map(|(g,u)|crate::bf(crate::bf_silu(g)*u)).collect();
    if !out.iter().all(|v|v.is_finite()) {return Err("MLP reuse output".into());}
    Ok((out,rg+ru))
}
pub(crate) fn evaluate_prepared<F,B>(r:&Request,q:&int8_kernel::QuantizedRows,ax:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    let requests=projections(r,m)?;
    output(&requests,q,ax,m,read)
}
pub(crate) fn evaluate<F,B>(r:&Request,x:&[f32],m:&Manifest,read:&mut F)->Result<(Vec<f32>,u64)>
where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer,
{
    let requests=projections(r,m)?;
    let (n,rows,cols,rank)=(r.dims[0],r.dims[1],r.dims[2],r.dims[4]);
    let q_len=n*cols;let sx_len=n*(cols/256);let state_len=q_len+sx_len+2*n*rank;
    let capture=r.op=="mlp_gate_up_capture";
    if state_len>crate::MAX_FLOATS || x.len()!=if capture {q_len} else {state_len}
        || (capture && n*rows+state_len>crate::MAX_FLOATS) || !x.iter().all(|v|v.is_finite()) {
        return Err("MLP reuse frame bounds".into());
    }
    if capture {
        let q=crate::profile::measure("activation_quantize",||int8_kernel::quantize_rows(x,n,cols))?;
        let mut ax=Vec::with_capacity(2*n*rank);let mut read_bytes=0;
        for request in &requests {
            let a=m.tensors.iter().find(|t|t.name==request.aux[0]).unwrap();
            let mut ar=request.clone();ar.op="matmul".into();ar.dims=vec![n,rank,cols];
            let (aw,bytes)=crate::load_prepared_weight(a,&ar,&mut *read)?;
            let values=crate::profile::measure("lora_matmul_A",||crate::matrix(x,&aw,n,rank,cols))?;
            ax.extend_from_slice(&values);read_bytes+=bytes;
        }
        let (mut out,bytes)=output(&requests,&q,&ax,m,read)?;
        q.append_wire_values(&mut out);out.extend_from_slice(&q.scales()[..sx_len]);out.extend_from_slice(&ax);
        if !out.iter().all(|v|v.is_finite()) {return Err("MLP reuse finite state".into());}
        Ok((out,bytes+read_bytes))
    } else {
        let q=int8_kernel::QuantizedRows::from_wire(n,cols,&x[..q_len],&x[q_len..q_len+sx_len])?;
        output(&requests,&q,&x[q_len+sx_len..],m,read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_original_a_products_reuse_without_rounding_and_bad_rank_is_rejected() {
        let mut data=Vec::new();let mut tensors=Vec::new();
        let gate="layer.mlp.gate_proj.weight";let up="layer.mlp.up_proj.weight";
        for (which,name) in [gate,up].into_iter().enumerate() {
            let prefix=name.strip_suffix(".weight").unwrap();
            for (name,rows,cols,int8) in [(name.to_string(),16,256,true),(format!("{prefix}.lora_A.weight"),4,256,false),(format!("{prefix}.lora_B.weight"),16,4,false)] {
                let offset=data.len() as u64;
                for i in 0..rows*cols {
                    if int8 {data.push(((i*17+which*11)%255) as u8);}
                    else {data.extend_from_slice(&(((i%19) as f32-9.+which as f32)/37.).to_le_bytes());}
                }
                if int8 {for i in 0..rows {data.extend_from_slice(&(0.0027+i as f32*0.0001).to_le_bytes());}}
                tensors.push(crate::Tensor {name,offset,rows,cols,dtype:if int8 {"int8"} else {"f32"}.into(),bytes:data.len() as u64-offset});
            }
        }
        let m=Manifest {version:1,model:"0".repeat(64),pack_hash:"1".repeat(64),bytes:data.len() as u64,tensors};
        for n in [1,7,8,32,132] {
            let x:Vec<_>=(0..n*256).map(|i|crate::bf(((i%29) as f32-14.)/11.)).collect();
            let mut r=Request {version:1,model:m.model.clone(),pack_hash:m.pack_hash.clone(),input_hash:"2".repeat(64),step:0,op:"mlp_gate_up_integer".into(),tensor:gate.into(),dims:vec![n,8,256,0],aux:vec![up.into()],scalars:vec![2.],encoding:"bf16-block256-exact-v1".into()};
            let mut read=|offset:u64,len:usize|Ok(data[offset as usize..offset as usize+len].to_vec());
            let original_first=crate::evaluate_with_reader(&r,&x,&m,&mut read).unwrap().0;
            r.op="mlp_gate_up_capture".into();r.encoding=crate::projection_codec::NAME.into();r.dims.push(4);
            let captured=crate::evaluate_with_reader(&r,&x,&m,&mut read).unwrap().0;
            assert_eq!(original_first.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),captured[..n*8].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let state=&captured[n*8..];
            let (_,roundtrip)=crate::decode(&crate::encode(&r,&captured).unwrap()).unwrap();
            assert_eq!(roundtrip.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),captured.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let mut ordinary=r.clone();ordinary.op="mlp_gate_up_integer".into();ordinary.encoding="bf16-block256-exact-v1".into();ordinary.dims=vec![n,8,256,8];
            let original_second=crate::evaluate_with_reader(&ordinary,&x,&m,&mut read).unwrap().0;
            r.op="mlp_gate_up_reuse".into();r.dims[3]=8;
            let packet=crate::encode(&r,state).unwrap();
            let (decoded,input)=crate::decode_query(&packet).unwrap();
            assert!(input.values().is_none());
            let reused=crate::evaluate_decoded_with_prepared_buffer(&decoded,&input,&m,&mut read).unwrap().0;
            assert_eq!(original_second.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),reused.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let generic=crate::evaluate_with_reader(&r,state,&m,&mut read).unwrap().0;
            assert_eq!(reused.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),generic.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let mut bad=r.clone();bad.dims[4]=3;
            assert!(evaluate(&bad,state,&m,&mut read).is_err());
            let mut bad_model=m.clone();
            bad_model.tensors[4].rows=3;
            assert!(evaluate(&r,state,&bad_model, &mut |_,_|->Result<Vec<u8>> {panic!("read before validating both sealed A shapes")}).is_err());
            let mut bad=r.clone();bad.aux[0]=gate.into();
            assert!(crate::evaluate_decoded_with_prepared_buffer(&bad,&input,&m,&mut read).is_err());
        }
    }
}
