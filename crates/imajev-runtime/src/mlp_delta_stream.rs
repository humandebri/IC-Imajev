//! MLP completion, next Delta head group and out-projection columns in one query.
use crate::{
    prepared_weights::{LoadedWeight, WeightBuffer},
    Manifest, Request, Result,
};
pub(crate) const NAME: &str = "mlp-delta-stream-exact-v1";
fn is_follow(r: &Request) -> bool {matches!(r.op.as_str(), "delta_partial_mlp_prepare" | "delta_partial_mlp_prepare_down" | "delta_partial_mlp_full" | "delta_partial_finish" | "delta_partial_mlp_front")}
const C: usize = 2560;
const H: usize = 9216;
const R: usize = 64;
fn plain(r:&Request)->bool {r.op=="mlp_full_delta_partial"}
fn metadata(r: &Request) -> Result<(usize, usize, usize, usize, usize, usize)> {
    if r.encoding != NAME
        || !matches!(
            r.op.as_str(),
            "mlp_complete_delta_partial" | "mlp_full_delta_partial" | "delta_partial_mlp_prepare" | "delta_partial_mlp_prepare_down" | "delta_partial_mlp_full" | "delta_partial_finish" | "delta_partial_mlp_front"
        )
        || r.dims.len() != if matches!(r.op.as_str(),"delta_partial_mlp_prepare_down"|"delta_partial_mlp_front") {6} else {5}
        || (r.op == "delta_partial_mlp_prepare_down" && !crate::mlp_pipeline::valid_partial_rows(*r.dims.get(5).unwrap_or(&0)))
        || (r.op=="delta_partial_mlp_front" && !(r.dims[5]>0 && r.dims[5]<H && r.dims[5]%crate::mlp_stream::STEP==0))
        || r.aux.len() != 1
        || r.scalars.len() != 2
        || r.scalars[0].to_bits() != 2f32.to_bits()
        || r.scalars[1].to_bits() != 1e-6f32.to_bits()
    {
        return Err("pair metadata".into());
    }
    let (n, b, c, h, p) = (r.dims[0], r.dims[1], r.dims[2], r.dims[3], r.dims[4]);
    if !(1..=89).contains(&n)
        || (if plain(r) {b!=0}else{b==0 && !is_follow(r)})
        || c == 0
        || b % crate::mlp_stream::STEP != 0
        || c % crate::mlp_stream::STEP != 0
        || b.checked_add(c) != Some(H)
        || h == 0
        || h >= 32
        || h % 2 != 0
        || p == 0
        || p > 132
    {
        return Err("pair bounds".into());
    }
    let s = r
        .tensor
        .strip_prefix("model.language_model.layers.")
        .and_then(|v| v.strip_suffix(".post_attention_layernorm.weight"))
        .ok_or("pair layer")?;
    let layer: usize = s.parse().map_err(|_| "pair layer")?;
    if layer >= 30
        || layer.to_string() != s
        || (layer + 2) % 4 == 0
        || r.aux[0]
            != format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            )
    {
        return Err("pair next Delta scope".into());
    }
    Ok((n, b, c, h, p, layer))
}
fn inner(r: &Request) -> Result<Request> {
    let (n, b, c, _, _, _) = metadata(r)?;
    let mut i = r.clone();
    if b==0 {i.op="mlp_full_integer".into();i.encoding="bf16-block256-exact-v1".into();i.dims=vec![n,C];return Ok(i);}
    i.encoding = crate::mlp_stream::NAME.into();
    i.op = "mlp_stream_complete".into();
    i.dims = vec![n, b, c];
    Ok(i)
}
pub(crate) fn reply_count(r: &Request) -> Result<usize> {
    let (n, _, _, h, _, _) = metadata(r)?;
    if r.op=="delta_partial_mlp_front" {return Ok(crate::mlp_stream::limit(&next_mlp(r)?)?+3*(32-h)*256);}
    Ok(if is_follow(r) {
        n * (if matches!(r.op.as_str(),"delta_partial_mlp_full"|"delta_partial_finish") {2*C}else{C+H+100}) + 3 * (32 - h) * 256
    } else {
        n * (2 * C + C / 256 + 2 * R + 64 + C + R) + 3 * h * 256
    })
}
// Dimensions are bounded by metadata; compressed descriptors never size allocations.
fn expand_residual(n:usize,begin:usize,p:&[u8])->Result<Vec<u8>> {
    if p.len()<9 || p[0]!=if begin%256==0{1}else{2} || u32::from_le_bytes(p[1..5].try_into().unwrap())as usize!=begin {return Err("pair residual progress".into());}
    let size=u32::from_le_bytes(p[5..9].try_into().unwrap())as usize;
    let end=9usize.checked_add(size).filter(|&end|end<=p.len()).ok_or("pair residual length")?;
    let tail=n*(C+begin+begin%256+4*(C/256+begin/256+3*R));
    if p.len()-end!=tail {return Err("pair residual tail".into());}
    let mut groups=crate::carry_planes::decode(&p[9..end],&[(n*C,2)])?;
    let planes=groups.pop().ok_or("pair residual planes")?;
    let mut out=Vec::with_capacity(5+2*n*C+tail);out.extend_from_slice(&p[..5]);
    out.resize(5+2*n*C,0);crate::carry_planes::interleave(&planes,&mut out[5..])?;
    out.extend_from_slice(&p[end..]);Ok(out)
}
/// Validated complete MLP input remains attached to the whole fusion request.
/// ```compile_fail
/// let mut x: imajev_runtime::PreparedMlpDeltaStream = todo!();
/// x.request.step += 1;
/// ```
pub struct PreparedMlpDeltaStream {
    request: Request,
    mlp_request: Request,
    mlp: Option<crate::PreparedMlpStream>,
    plain: Option<Vec<f32>>,
    follow: Option<(
        Vec<f32>,
        crate::delta_head_continue::Preparation,
        Vec<f32>,
        Vec<f32>,
    )>,
    history: Vec<f32>,
    log: Vec<f32>,
}
impl PreparedMlpDeltaStream {
    pub(crate) fn decode(r: &Request, payload: &[u8]) -> Result<Self> {
        let (n, begin, _, h, p, _) = metadata(r)?;
        if is_follow(r) {
            return Self::decode_follow(r, payload, n, h, p);
        }
        if !(if plain(r){payload.first()==Some(&13)}else{matches!(payload.first(),Some(&1)|Some(&6))}) || payload.len() < 5 {
            return Err("pair request direction".into());
        }
        let len = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
        let end = 5usize
            .checked_add(len)
            .filter(|end| *end <= payload.len())
            .ok_or("pair MLP length")?;
        let hc = 3 * h * 256;
        let kc = p * h / 2 * 128;
        let tail = p * (h * 128 + h);
        if payload.len() != end + 2 * (hc + kc) + 4 * tail {
            return Err("pair prefix length".into());
        }
        let mlp_request = inner(r)?;
        let expanded;
        let carry=if payload[0]==6 {
            expanded=crate::profile::measure("pair_residual_planes",||expand_residual(n,begin,&payload[5..end]))?;
            &expanded[..]
        }else{&payload[5..end]};
        let (mlp,plain_values)=if plain(r) {
            if carry.len()!=4*n*C{return Err("pair full MLP input shape".into());}
            let mut v=vec![0.;2*n*C];crate::bf16_codec::unpack(carry,&mut v);
            if !v.iter().all(|v|v.is_finite()){return Err("pair full MLP finite".into());}(None,Some(v))
        }else{(Some(crate::PreparedMlpStream::decode(&mlp_request, carry)?),None)};
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..end + 2 * hc], &mut history);
        let mut log = vec![0.; kc];
        crate::bf16_codec::unpack(&payload[end + 2 * hc..end + 2 * (hc + kc)], &mut log);
        log.extend(
            payload[end + 2 * (hc + kc)..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        if !history.iter().chain(&log).all(|v| v.is_finite())
            || !log[kc + p * h * 128..]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        {
            return Err("pair prefix finite/decay".into());
        }
        Ok(Self {
            request: r.clone(),
            mlp_request,
            mlp,
            plain: plain_values,
            follow: None,
            history,
            log,
        })
    }
    fn decode_follow(r: &Request, payload: &[u8], n: usize, h: usize, p: usize) -> Result<Self> {
        if matches!(payload.first(),Some(&12)|Some(&14)) {
            if payload.len()<9 {return Err("pair hidden length".into());}
            // Decode only the already quantized byte column; no scale or float
            // changes. Descriptor lengths never determine output allocations.
            let input_planes;
            let body=if payload[0]==14 {
                let len=u32::from_le_bytes(payload[1..5].try_into().unwrap())as usize;
                let end=5usize.checked_add(len).filter(|v|v.checked_add(8).is_some_and(|v|v<=payload.len())).ok_or("pair input plane range")?;
                input_planes=crate::profile::measure("pair_input_planes",||crate::carry_planes::decode(&payload[5..end],&[(n*C,1)]))?;
                &payload[end..]
            }else{input_planes=vec![];&payload[1..]};
            let len=u32::from_le_bytes(body[..4].try_into().unwrap())as usize;
            let end=4usize.checked_add(len).filter(|v|v.checked_add(4).is_some_and(|v|v<=body.len())).ok_or("pair hidden range")?;
            let planes=crate::profile::measure("pair_hidden_planes",||crate::carry_planes::decode(&body[4..end],&[(n*C,2)]))?;
            let mut plain=Vec::with_capacity(5+2*n*C+body.len()-end-4+if payload[0]==14{n*C}else{0});plain.push(10);plain.extend_from_slice(&body[end..end+4]);
            plain.resize(5+2*n*C,0);crate::carry_planes::interleave(&planes[0],&mut plain[5..])?;
            if payload[0]==14{plain.extend_from_slice(&input_planes[0]);}
            plain.extend_from_slice(&body[end+4..]);
            return Self::decode_compressed_follow(r,&plain,n,h,p);
        }
        if matches!(payload.first(),Some(&4)|Some(&10)) {return Self::decode_compressed_follow(r,payload,n,h,p);}
        let remaining = 32 - h;
        let hc = 3 * remaining * 256;
        let kc = p * remaining / 2 * 128;
        let prefix_tail = p * (remaining * 128 + remaining);
        let tail = n * (C / 256 + 2 * R + 64 + C + R);
        let end = 1 + 3 * n * C + 4 * tail;
        if payload.first() != Some(&2) || payload.len() != end + 2 * (hc + kc) + 4 * prefix_tail {
            return Err("pair continuation length/direction".into());
        }
        let mut hidden = vec![0.; n * C];
        crate::bf16_codec::unpack(&payload[1..1 + 2 * n * C], &mut hidden);
        let ints = &payload[1 + 2 * n * C..1 + 3 * n * C];
        let rest: Vec<_> = payload[1 + 3 * n * C..end]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if !hidden.iter().chain(&rest).all(|v| v.is_finite()) {
            return Err("pair continuation finite".into());
        }
        let prep_end = n * (C / 256 + 2 * R + 64);
        let prep = crate::delta_head_continue::Preparation::restore(n, ints, &rest[..prep_end])?;
        let base = rest[prep_end..prep_end + n * C].to_vec();
        let ax = rest[prep_end + n * C..].to_vec();
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..end + 2 * hc], &mut history);
        let mut log = vec![0.; kc];
        crate::bf16_codec::unpack(&payload[end + 2 * hc..end + 2 * (hc + kc)], &mut log);
        log.extend(
            payload[end + 2 * (hc + kc)..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
        if !history.iter().chain(&log).all(|v| v.is_finite())
            || !log[kc + p * remaining * 128..]
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        {
            return Err("pair continuation prefix".into());
        }
        Ok(Self {
            request: r.clone(),
            mlp_request: inner(r)?,
            mlp: None,
            plain: None,
            follow: Some((hidden, prep, base, ax)),
            history,
            log,
        })
    }
    fn decode_compressed_follow(r:&Request,payload:&[u8],n:usize,h:usize,p:usize)->Result<Self> {
        if payload.len()<5 {return Err("pair compressed length".into());}
        let len=u32::from_le_bytes(payload[1..5].try_into().unwrap())as usize;
        let prep=n*(C/256+2*R+64);let end=5+3*n*C+4*(prep+n*R);
        let remaining=32-h;let prefix=2*(3*remaining*256+p*remaining/2*128)+4*p*(remaining*128+remaining);
        let packed_end=end.checked_add(len).filter(|v|if payload[0]==10 {*v<=payload.len()}else{v.checked_add(prefix)==Some(payload.len())}).ok_or("pair compressed range")?;
        let planes=crate::profile::measure("pair_base_planes",||crate::carry_planes::decode(&payload[end..packed_end],&[(n*C,4)]))?;
        let mut plain=Vec::with_capacity(1+3*n*C+4*(prep+n*C+n*R)+prefix);plain.push(2);
        plain.extend_from_slice(&payload[5..5+3*n*C+4*prep]);
        for index in 0..n*C {for byte in 0..4 {plain.push(planes[0][byte*n*C+index]);}}
        plain.extend_from_slice(&payload[5+3*n*C+4*prep..end]);
        if payload[0]==10 {
            let shapes=[(3*remaining*256,2),(p*remaining/2*128,2),(p*(remaining*128+remaining),4)];
            let groups=crate::profile::measure("pair_prefix_planes",||crate::carry_planes::decode(&payload[packed_end..],&shapes))?;
            for (group,(count,width)) in groups.iter().zip(shapes) {
                let start=plain.len();plain.resize(start+count*width,0);
                if width==2 {crate::carry_planes::interleave(group,&mut plain[start..])?;}
                else {for i in 0..count {for b in 0..width {plain[start+i*width+b]=group[b*count+i];}}}
            }
        }else{plain.extend_from_slice(&payload[packed_end..]);}
        Self::decode_follow(r,&plain,n,h,p)
    }
    fn evaluate_follow<F, B>(
        self,
        r: &Request,
        m: &Manifest,
        read: &mut F,
        n: usize,
        h: usize,
        p: usize,
        layer: usize,
    ) -> Result<(Vec<f32>, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        let (hidden, prep, base, ax) = self.follow.ok_or("pair continuation direction")?;
        let remaining = 32 - h;
        let count = remaining * 128;
        let mut dr = r.clone();
        dr.op = "delta_project_reuse".into();
        dr.encoding = "bf16-block256-exact-v1".into();
        dr.tensor = format!(
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
            layer + 1
        );
        dr.aux.clear();
        dr.scalars.clear();
        dr.dims = vec![n, remaining, h, 0];
        let (gated, history, mut bytes) = crate::profile::measure("pair_remaining_heads", || {
            crate::delta_head_continue::group(
                &dr,
                &prep,
                &self.history,
                &self.log,
                n,
                remaining,
                h,
                p,
                m,
                read,
            )
        })?;
        let root = dr.tensor.strip_suffix(".in_proj_qkv.weight").unwrap();
        let mut kr = dr.clone();
        kr.op = "linear_integer_k_continue".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.dims = vec![n, C, 4096, h * 128, count];
        let mut input = gated.clone();
        input.extend(base);
        let (base, used) = crate::int8_k_continue::evaluate(&kr, &input, m, read)?;
        bytes += used;
        drop(input);
        let aname = format!("{root}.out_proj.lora_A.weight");
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == aname)
            .ok_or("pair continuation A")?;
        if a.dtype != "f32" || a.rows != R || a.cols != 4096 || a.bytes != (R * 4096 * 4) as u64 {
            return Err("pair continuation A shape".into());
        }
        kr.op = "matmul".into();
        kr.tensor = aname;
        kr.dims = vec![n, R, 4096];
        let (w, used) = crate::load_prepared_weight(a, &kr, &mut *read)?;
        bytes += used;
        let LoadedWeight::Prepared(w) = w else {
            return Err("pair continuation fixed A".into());
        };
        let ax = crate::f32_output::continue_columns(&gated, &ax, &w, n, R, 4096, h * 128, count)?;
        drop(w);
        kr.op = "linear_integer_k_finish".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.dims = vec![n, C, 4096];
        kr.scalars = vec![2.];
        kr.aux = vec![
            format!("{root}.out_proj.lora_A.weight"),
            format!("{root}.out_proj.lora_B.weight"),
        ];
        let mut state = base;
        state.extend(ax);
        let (attention, used) = crate::int8_k_continue::evaluate(&kr, &state, m, read)?;
        bytes += used;
        drop(state);
        let mut pair = hidden;
        pair.extend(attention);
        if r.op == "delta_partial_finish" {pair.extend(history);return Ok((pair, bytes));}
        let mr = next_mlp(r)?;
        let (mut out, used) = crate::profile::measure("pair_next_mlp_prepare", || {
            if mr.op=="mlp_stream_prepare" {crate::mlp_stream::prepare_direct(&mr,pair,m,read)}
            else if mr.op == "mlp_full_integer" {
                crate::mlp_pipeline::full(&mr,&pair,m,read)
            } else if mr.op == "mlp_prepare_partial_down" {
                crate::mlp_pipeline::prepare_partial(&mr, &pair, m, read)
            } else {crate::mlp_pipeline::prepare(&mr, &pair, m, read)}
        })?;
        bytes += used;
        out.extend(history);
        Ok((out, bytes))
    }
    pub(crate) fn evaluate<F, B>(
        self,
        r: &Request,
        m: &Manifest,
        read: &mut F,
    ) -> Result<(Vec<f32>, u64)>
    where
        F: FnMut(u64, usize) -> Result<B>,
        B: WeightBuffer,
    {
        self.evaluate_inner(r,m,read,false).and_then(|(reply,bytes)|Ok((reply.into_values()?,bytes)))
    }
    #[cfg(feature="experimental-direct-mlp-reply")]
    pub(crate) fn evaluate_reply<F,B>(self,r:&Request,m:&Manifest,read:&mut F)->Result<(crate::EvaluatedReply,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        self.evaluate_inner(r,m,read,true)
    }
    fn evaluate_inner<F,B>(self,r:&Request,m:&Manifest,read:&mut F,direct:bool)->Result<(crate::EvaluatedReply,u64)>
    where F:FnMut(u64,usize)->Result<B>,B:WeightBuffer {
        if !crate::same_request(r, &self.request) {
            return Err("pair identity".into());
        }
        let (n, _, _, h, p, layer) = metadata(r)?;
        if is_follow(r) {
            return self.evaluate_follow(r, m, read, n, h, p, layer).map(|(v,b)|(crate::EvaluatedReply::Values(v),b));
        }
        let (both, mut bytes) = crate::profile::measure("pair_mlp_complete", || {
            if let Some(v)=self.plain {return crate::mlp_pipeline::full(&self.mlp_request,&v,m,read);}
            self.mlp
                .ok_or("pair completion direction")?
                .evaluate(&self.mlp_request, m, read)
        })?;
        let mut dr = r.clone();
        dr.op = "delta_project_capture".into();
        dr.encoding = "bf16-block256-exact-v1".into();
        dr.tensor = format!(
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
            layer + 1
        );
        dr.aux.clear();
        dr.scalars.clear();
        dr.dims = vec![n, h, 0, 0];
        let (prep, used) =
            crate::delta_head_continue::Preparation::capture(&dr, &both[n * C..], m, read)?;
        bytes += used;
        let (gated, history, used) = crate::profile::measure("pair_delta_heads", || {
            crate::delta_head_continue::group(
                &dr,
                &prep,
                &self.history,
                &self.log,
                n,
                h,
                0,
                p,
                m,
                read,
            )
        })?;
        bytes += used;
        let root = dr.tensor.strip_suffix(".in_proj_qkv.weight").unwrap();
        let count = h * 128;
        let mut kr = r.clone();
        kr.encoding = "bf16-block256-exact-v1".into();
        kr.op = "linear_integer_k_continue".into();
        kr.tensor = format!("{root}.out_proj.weight");
        kr.aux.clear();
        kr.scalars.clear();
        kr.dims = vec![n, C, 4096, 0, count];
        let mut input = gated.clone();
        input.extend(vec![0.; n * C]);
        let (base, used) = crate::int8_k_continue::evaluate(&kr, &input, m, read)?;
        bytes += used;
        drop(input);
        let aname = format!("{root}.out_proj.lora_A.weight");
        let a = m
            .tensors
            .iter()
            .find(|t| t.name == aname)
            .ok_or("pair out A")?;
        if a.dtype != "f32" || a.rows != R || a.cols != 4096 || a.bytes != (R * 4096 * 4) as u64 {
            return Err("pair out A shape".into());
        }
        kr.op = "matmul".into();
        kr.tensor = aname;
        kr.dims = vec![n, R, 4096];
        let (w, used) = crate::load_prepared_weight(a, &kr, &mut *read)?;
        bytes += used;
        let LoadedWeight::Prepared(w) = w else {
            return Err("pair fixed F32 A".into());
        };
        let ax = crate::profile::measure("pair_out_A_continue", || {
            crate::f32_output::continue_columns(&gated, &vec![0.; n * R], &w, n, R, 4096, 0, count)
        })?;
        #[cfg(feature="experimental-direct-mlp-reply")]
        if direct {
            let payload=direct_pair_payload(r,&both[..n*C],&prep,&base,&ax,&history)?;
            return Ok((crate::EvaluatedReply::payload(r,payload),bytes));
        }
        #[cfg(not(feature="experimental-direct-mlp-reply"))]
        let _=direct;
        let mut out = both[..n * C].to_vec();
        prep.flat(n, &mut out);
        out.extend(base);
        out.extend(ax);
        out.extend(history);
        if out.len() != reply_count(r)? || !out.iter().all(|v| v.is_finite()) {
            return Err("pair output length/finite".into());
        }
        Ok((crate::EvaluatedReply::Values(out), bytes))
    }
}
// QuantizedRows can only originate in the checked decoder or quantizer. Keep
// its integer representation throughout reply construction; never flatten it
// to F32 only to validate and pack the same integers again.
#[cfg(feature="experimental-direct-mlp-reply")]
fn direct_pair_payload(r:&Request,hidden:&[f32],prep:&crate::delta_head_continue::Preparation,base:&[f32],ax:&[f32],history:&[f32])->Result<Vec<u8>> {
    let(n,_,_,h,_,_)=metadata(r)?;
    if is_follow(r) || hidden.len()!=n*C || base.len()!=n*C || ax.len()!=n*R || history.len()!=3*h*256
        || prep.q.rows()!=n || prep.q.cols()!=C || prep.qa.len()!=n*R || prep.za.len()!=n*R || prep.gates.len()!=n*64
        || !crate::bf16_codec::classify_finite(hidden)? || !crate::bf16_codec::classify_finite(history)?
        || !prep.qa.iter().chain(&prep.za).chain(base).chain(ax).all(|v|v.is_finite())
        || !prep.gates.iter().all(|v|v.is_finite() && (0.0..=1.0).contains(v)) {
        return Err("direct pair reply invariant".into());
    }
    let mut payload=Vec::with_capacity(1+3*n*C+4*n*(C/256+3*R+64+C)+2*history.len());
    payload.push(0);
    let start=payload.len();payload.resize(start+2*hidden.len(),0);crate::bf16_codec::pack(hidden,&mut payload[start..]);
    payload.extend(prep.q.values()[..n*C].iter().map(|v|*v as i8 as u8));
    for v in prep.q.scales()[..n*C/256].iter().chain(&prep.qa).chain(&prep.za).chain(&prep.gates).chain(base).chain(ax) {
        payload.extend(v.to_le_bytes());
    }
    let start=payload.len();payload.resize(start+2*history.len(),0);crate::bf16_codec::pack(history,&mut payload[start..]);
    Ok(payload)
}
fn next_mlp(r: &Request) -> Result<Request> {
    let (n, _, _, _, _, layer) = metadata(r)?;
    let mut mr = r.clone();
    mr.op = "mlp_prepare_down".into();
    mr.encoding = crate::mlp_pipeline::NAME.into();
    mr.tensor = format!(
        "model.language_model.layers.{}.post_attention_layernorm.weight",
        layer + 1
    );
    mr.aux = vec![format!(
        "model.language_model.layers.{}.input_layernorm.weight",
        layer + 2
    )];
    mr.dims = vec![n, C];
    if r.op=="delta_partial_mlp_full" {mr.op="mlp_full_integer".into();mr.encoding="bf16-block256-exact-v1".into();}
    if r.op == "delta_partial_mlp_prepare_down" {
        mr.op = "mlp_prepare_partial_down".into();
        mr.dims.push(r.dims[5]);
    }
    if r.op=="delta_partial_mlp_front" {mr.op="mlp_stream_prepare".into();mr.encoding=crate::mlp_stream::NAME.into();mr.dims=vec![n,0,r.dims[5]];}
    Ok(mr)
}
fn pack_bf(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
    if !crate::bf16_codec::all_bf16(x) {
        return Err("pair BF16 reply".into());
    }
    let start = b.len();
    b.resize(start + 2 * x.len(), 0);
    crate::bf16_codec::pack(x, &mut b[start..]);
    Ok(())
}
pub(crate) fn append_reply(b: &mut Vec<u8>, r: &Request, x: &[f32]) -> Result<()> {
    let (n, _, _, h, _, _) = metadata(r)?;
    if x.len() != reply_count(r)? || !x.iter().all(|v| v.is_finite()) {
        return Err("pair reply length/finite".into());
    }
    if matches!(r.op.as_str(),"delta_partial_mlp_full"|"delta_partial_finish") {b.push(if r.op=="delta_partial_finish"{8}else{5});return pack_bf(b,x);}
    if r.op=="delta_partial_mlp_front" {let end=x.len()-3*(32-h)*256;b.push(9);crate::mlp_stream::append(b,&next_mlp(r)?,&x[..end])?;return pack_bf(b,&x[end..]);}
    if is_follow(r) {
        b.push(3);
        let c = n * (C + H + 100);
        crate::mlp_pipeline::append(b, &next_mlp(r)?, &x[..c])?;
        return pack_bf(b, &x[c..]);
    }
    let mut cursor = 0;
    b.push(0);
    fn bf(b: &mut Vec<u8>, x: &[f32]) -> Result<()> {
        if !crate::bf16_codec::all_bf16(x) {
            return Err("pair BF16 reply".into());
        }
        let start = b.len();
        b.resize(start + 2 * x.len(), 0);
        crate::bf16_codec::pack(x, &mut b[start..]);
        Ok(())
    }
    bf(b, &x[..n * C])?;
    cursor += n * C;
    let start = b.len();
    b.resize(start + n * C, 0);
    crate::projection_codec::pack_integers(&x[cursor..cursor + n * C], &mut b[start..])?;
    cursor += n * C;
    let tail = n * (C / 256 + 2 * R + 64 + C + R);
    if x[cursor..cursor + n * C / 256].iter().any(|v| *v <= 0.)
        || x[cursor + n * (C / 256 + 2 * R)..cursor + n * (C / 256 + 2 * R + 64)]
            .iter()
            .any(|v| !(0.0..=1.0).contains(v))
    {
        return Err("pair reply scales/gates".into());
    }
    for v in &x[cursor..cursor + tail] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    cursor += tail;
    bf(b, &x[cursor..cursor + 3 * h * 256])?;
    Ok(())
}
pub(crate) fn decode_reply(r: &Request, payload: &[u8]) -> Result<Vec<f32>> {
    let (n, _, _, h, _, _) = metadata(r)?;
    if matches!(r.op.as_str(),"delta_partial_mlp_full"|"delta_partial_finish") {
        if payload.first()!=Some(&if r.op=="delta_partial_finish"{8}else{5})||payload.len()!=1+2*reply_count(r)?{return Err("pair full reply length/direction".into());}
        let mut out=vec![0.;reply_count(r)?];crate::bf16_codec::unpack(&payload[1..],&mut out);if !out.iter().all(|v|v.is_finite()){return Err("pair full reply finite".into());}return Ok(out);
    }
    if r.op=="delta_partial_mlp_front" {let tail=2*3*(32-h)*256;if payload.first()!=Some(&9)||payload.len()<1+tail{return Err("pair front reply length/direction".into());}let end=payload.len()-tail;let mut out=crate::mlp_stream::decode_values(&next_mlp(r)?,&payload[1..end])?;let mut history=vec![0.;3*(32-h)*256];crate::bf16_codec::unpack(&payload[end..],&mut history);out.extend(history);if out.len()!=reply_count(r)?||!out.iter().all(|v|v.is_finite()){return Err("pair front reply finite/count".into());}return Ok(out);}
    if is_follow(r) {
        let end = 2 + n * (C * 2 + H + 400);
        let hc = 3 * (32 - h) * 256;
        if payload.first() != Some(&3) || payload.len() != end + 2 * hc {
            return Err("pair prepared reply direction/length".into());
        }
        let mut out = crate::mlp_pipeline::decode_values(&next_mlp(r)?, &payload[1..end])?;
        let mut history = vec![0.; hc];
        crate::bf16_codec::unpack(&payload[end..], &mut history);
        if !history.iter().all(|v| v.is_finite()) {
            return Err("pair history finite".into());
        }
        out.extend(history);
        return Ok(out);
    }
    let bf_count = n * C;
    let tail = n * (C / 256 + 2 * R + 64 + C + R);
    if payload.first() != Some(&0) || payload.len() != 1 + bf_count * 3 + tail * 4 + 3 * h * 256 * 2
    {
        return Err("pair reply length/direction".into());
    }
    let mut out = vec![0.; bf_count];
    crate::bf16_codec::unpack(&payload[1..1 + bf_count * 2], &mut out);
    let mut cursor = 1 + bf_count * 2;
    if payload[cursor..cursor + bf_count].contains(&128) {
        return Err("pair integer reply".into());
    }
    out.extend(
        payload[cursor..cursor + bf_count]
            .iter()
            .map(|b| *b as i8 as f32),
    );
    cursor += bf_count;
    out.extend(
        payload[cursor..cursor + tail * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
    );
    cursor += tail * 4;
    let mut history = vec![0.; 3 * h * 256];
    crate::bf16_codec::unpack(&payload[cursor..], &mut history);
    out.extend(history);
    let mut checked = vec![];
    append_reply(&mut checked, r, &out)?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request(n: usize, h: usize) -> Request {
        serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"mlp_complete_delta_partial","encoding":NAME,"tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,4352,4864,h,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()
    }
    #[test]
    fn maximum_reply_is_exact_bounded_and_not_a_query_input() {
        for n in [1, 7, 87, 89] {
            for h in [22, 24] {
                let r = request(n, h);
                let mut x = vec![-0.; n * C];
                x.extend(vec![-127.; n * C]);
                x.extend(vec![0.0123; n * C / 256]);
                x.extend(vec![0.1234567; n * 2 * R]);
                x.extend(vec![0.875; n * 64]);
                x.extend(vec![-0.234567; n * C]);
                x.extend(vec![0.345678; n * R]);
                x.extend(vec![0.5; 3 * h * 256]);
                let frame = crate::encode(&r, &x).unwrap();
                assert!(frame.len() < 2_000_000);
                let (_, y) = crate::decode(&frame).unwrap();
                assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(crate::decode_query(&frame).is_err());
                for offset in [2 * n * C, 2 * n * C + n * (C / 256 + 2 * R)] {
                    let mut bad = x.clone();
                    bad[offset] = f32::NAN;
                    assert!(crate::encode(&r, &bad).is_err());
                }
            }
        }
    }
    fn input(r: &Request) -> Vec<u8> {
        let (n, b, _, h, p, _) = metadata(r).unwrap();
        let mr = inner(r).unwrap();
        let mut x = vec![0.; n * 2 * C];
        x.extend(vec![1.; n * C / 256]);
        x.extend(vec![0.; n * 2 * R]);
        x.extend(vec![0.; n * b]);
        x.extend(vec![1.; n * b / 256]);
        x.extend(vec![0.; n * R]);
        let frame = crate::encode(&mr, &x).unwrap();
        let hlen = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        let part = &frame[4 + hlen..frame.len() - 32];
        let mut payload = vec![1];
        payload.extend((part.len() as u32).to_le_bytes());
        payload.extend(part);
        payload.extend(vec![0u8; 2 * (3 * h * 256 + p * h / 2 * 128)]);
        for _ in 0..p * (h * 128 + h) {
            payload.extend(0f32.to_le_bytes());
        }
        payload
    }
    #[test]
    fn continuation_direction_identity_and_large_prepared_reply_are_checked() {
        for n in [1, 89] {
            let mut r = request(n, 22);
            let mut x = vec![0.; n * 2 * C];
            x.extend(vec![1.; n * C / 256]);
            x.extend(vec![0.1234567; n * 2 * R]);
            x.extend(vec![0.875; n * 64]);
            x.extend(vec![0.234567; n * C]);
            x.extend(vec![0.345678; n * R]);
            x.extend(vec![0.; 3 * 22 * 256]);
            let mut encoded = vec![];
            append_reply(&mut encoded, &r, &x).unwrap();
            encoded.truncate(encoded.len() - 2 * 3 * 22 * 256);
            encoded[0] = 2;
            let remaining = 10;
            encoded.extend(vec![
                0;
                2 * (3 * remaining * 256 + 45 * remaining / 2 * 128)
            ]);
            for _ in 0..45 * (remaining * 128 + remaining) {
                encoded.extend(0f32.to_le_bytes());
            }
            r.op = "delta_partial_mlp_prepare".into();
            let header_bytes = serde_json::to_vec(&r).unwrap().len();
            assert!(encoded.len() + header_bytes + 36 < 2_000_000);
            let prepared = PreparedMlpDeltaStream::decode(&r, &encoded).unwrap();
            let m = Manifest {
                version: 1,
                model: r.model.clone(),
                pack_hash: r.pack_hash.clone(),
                bytes: 0,
                tensors: vec![],
            };
            let mut bad = r.clone();
            bad.step += 1;
            assert_eq!(
                prepared
                    .evaluate(&bad, &m, &mut |_, _| -> Result<Vec<u8>> {
                        panic!("follow identity read")
                    })
                    .unwrap_err(),
                "pair identity"
            );
            let mut invalid = encoded.clone();
            invalid[1 + 2 * n * C] = 128;
            assert!(PreparedMlpDeltaStream::decode(&r, &invalid).is_err());
            invalid = encoded.clone();
            invalid[1 + 3 * n * C..1 + 3 * n * C + 4].copy_from_slice(&0f32.to_le_bytes());
            assert!(PreparedMlpDeltaStream::decode(&r, &invalid).is_err());
            let mut reply = vec![-0.; n * C];
            reply.extend(vec![127.; n * H]);
            reply.extend(vec![0.009; n * 36]);
            reply.extend(vec![-0.1234567; n * R]);
            reply.extend(vec![0.; 3 * remaining * 256]);
            let frame = crate::encode(&r, &reply).unwrap();
            assert!(frame.len() < 2_000_000);
            let (_, actual) = crate::decode(&frame).unwrap();
            assert!(reply
                .iter()
                .zip(actual)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(crate::decode_query(&frame).is_err());
        }
    }
    #[test]
    fn compressed_base_follow_decodes_and_rejects_trailing_planes() {
        let mut r=request(1,20);r.op="delta_partial_mlp_prepare".into();
        let prep=C/256+2*R+64;let mut payload=vec![4];let packed_len=4*(5+C);
        payload.extend((packed_len as u32).to_le_bytes());payload.extend(vec![0;3*C]);
        for _ in 0..C/256 {payload.extend(1f32.to_le_bytes());}
        payload.extend(vec![0;4*(prep-C/256+R)]);
        for _ in 0..4 {payload.push(0);payload.extend((C as u32).to_le_bytes());payload.extend(vec![0;C]);}
        let remain=12;payload.extend(vec![0;2*(3*remain*256+45*remain/2*128)+4*45*(remain*128+remain)]);
        assert!(PreparedMlpDeltaStream::decode(&r,&payload).is_ok());
        payload.push(0);assert!(PreparedMlpDeltaStream::decode(&r,&payload).is_err());
        payload[1..5].copy_from_slice(&u32::MAX.to_le_bytes());assert!(PreparedMlpDeltaStream::decode(&r,&payload).is_err());
    }
    #[test]
    fn partial_down_progress_is_explicit_and_bound() {
        let mut r = request(1, 22);
        r.op = "delta_partial_mlp_prepare_down".into();
        r.dims.push(1600);
        assert!(metadata(&r).is_ok());
        let next = next_mlp(&r).unwrap();
        assert_eq!(next.op, "mlp_prepare_partial_down");
        assert_eq!(next.dims, vec![1, C, 1600]);
        for rows in [0, 31, C, usize::MAX] {
            r.dims[5] = rows;
            assert!(metadata(&r).is_err());
        }
        r.dims[5] = 1600;
        r.op = "delta_partial_mlp_prepare".into();
        assert!(metadata(&r).is_err());
    }
    #[test]
    fn malformed_metadata_prefix_and_identity_fail_before_reads() {
        let r = request(1, 24);
        let payload = input(&r);
        let m = Manifest {
            version: 1,
            model: r.model.clone(),
            pack_hash: r.pack_hash.clone(),
            bytes: 0,
            tensors: vec![],
        };
        let state = PreparedMlpDeltaStream::decode(&r, &payload).unwrap();
        let mut changed = r.clone();
        changed.step += 1;
        assert_eq!(
            state
                .evaluate(&changed, &m, &mut |_, _| -> Result<Vec<u8>> {
                    panic!("identity read")
                })
                .unwrap_err(),
            "pair identity"
        );
        for dims in [
            vec![],
            vec![0, 4352, 4864, 24, 45],
            vec![90, 4352, 4864, 24, 45],
            vec![1, 0, 9216, 24, 45],
            vec![1, 4352, 4864, 23, 45],
            vec![1, 4352, 4864, 32, 45],
            vec![1, 4352, 4864, 24, 0],
            vec![1, 4352, 4864, 24, 133],
            vec![1, usize::MAX, 256, 24, 45],
        ] {
            let mut bad = r.clone();
            bad.dims = dims;
            assert!(PreparedMlpDeltaStream::decode(&bad, &payload).is_err());
        }
        for value in [f32::NAN, 1.1, -1.] {
            let mut bad = payload.clone();
            let last = bad.len() - 4;
            bad[last..].copy_from_slice(&value.to_le_bytes());
            assert!(PreparedMlpDeltaStream::decode(&r, &bad).is_err());
        }
        let mut bad = payload.clone();
        bad.push(0);
        assert!(PreparedMlpDeltaStream::decode(&r, &bad).is_err());
        assert!(PreparedMlpDeltaStream::decode(&r, &payload[..payload.len() - 1]).is_err());
    }
}
#[cfg(test)]mod full_follow_tests {
 use super::*;
 #[test]fn residual_planes_preserve_bytes_and_reject_malformed(){
  let(n,begin)=(1,256);let tail=n*(C+begin+begin%256+4*(C/256+begin/256+3*R));
  let mut planes=vec![];for byte in [0u8,128] {planes.push(2);planes.extend(1u32.to_le_bytes());planes.push(byte);}
  let mut p=vec![1];p.extend((begin as u32).to_le_bytes());p.extend((planes.len()as u32).to_le_bytes());p.extend(planes);p.extend(vec![17;tail]);
  let mut expected=vec![1];expected.extend((begin as u32).to_le_bytes());expected.extend(vec![0,128].repeat(n*C));expected.extend(vec![17;tail]);
  assert_eq!(expand_residual(n,begin,&p).unwrap(),expected);
  let mut bad=p.clone();bad[1]=1;assert!(expand_residual(n,begin,&bad).is_err());
  bad=p.clone();bad[5..9].copy_from_slice(&u32::MAX.to_le_bytes());assert!(expand_residual(n,begin,&bad).is_err());
  bad=p.clone();bad.push(0);assert!(expand_residual(n,begin,&bad).is_err());p.pop();assert!(expand_residual(n,begin,&p).is_err());
 }
 #[test]fn full_reply_preserves_bits_and_rejects_wrong_direction(){
  let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":0,"op":"delta_partial_mlp_full","encoding":NAME,"tensor":"model.language_model.layers.3.post_attention_layernorm.weight","dims":[1,5376,3840,24,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.4.input_layernorm.weight"]})).unwrap();
  let v=vec![-0.;reply_count(&r).unwrap()];let frame=crate::encode(&r,&v).unwrap();let(_,decoded)=crate::decode(&frame).unwrap();assert_eq!(v.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),decoded.iter().map(|v|v.to_bits()).collect::<Vec<_>>());let mut raw=vec![];append_reply(&mut raw,&r,&v).unwrap();raw[0]=3;assert!(decode_reply(&r,&raw).is_err());assert_eq!(next_mlp(&r).unwrap().op,"mlp_full_integer");
 }
}

#[cfg(all(test,feature="experimental-direct-mlp-reply"))]
mod direct_reply_tests {
    use super::*;
    fn request(n:usize,h:usize)->Request {
        serde_json::from_value(serde_json::json!({"version":3,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"mlp_complete_delta_partial","encoding":NAME,"tensor":"model.language_model.layers.0.post_attention_layernorm.weight","dims":[n,4096,5120,h,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.1.input_layernorm.weight"]})).unwrap()
    }
    fn preparation(n:usize)->crate::delta_head_continue::Preparation {
        let integers=(0..n*C).map(|i|((i%255)as i16-127)as i8 as u8).collect::<Vec<_>>();
        let mut rest=vec![0.0123456;n*C/256];rest.extend(vec![0.1234567;n*2*R]);rest.extend(vec![0.5;n*64]);
        crate::delta_head_continue::Preparation::restore(n,&integers,&rest).unwrap()
    }
    #[test]
    fn typed_pair_matches_old_payload_and_authenticated_frames() {
        for n in [1,7,80,87,89] {for h in [18,20,26] {
            let mut r=request(n,h);let prep=preparation(n);
            let hidden=vec![-0.;n*C];let base=vec![0.1234567;n*C];let ax=vec![-0.1234567;n*R];let history=vec![-0.;3*h*256];
            let mut values=hidden.clone();prep.flat(n,&mut values);values.extend(&base);values.extend(&ax);values.extend(&history);
            let payload=direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).unwrap();let mut old=vec![];append_reply(&mut old,&r,&values).unwrap();assert_eq!(payload,old);
            for version in [1,2,3] {
                if version==2 && !cfg!(feature="experimental-blake3") {continue;}
                r.version=version;let mut output=r.clone();output.step+=1;
                for signed in [false,true] {
                    let old=if signed && cfg!(feature="experimental-host-checksum") {crate::encode_impl(&output,&values,true).unwrap()}else{crate::encode(&output,&values).unwrap()};
                    assert_eq!(crate::EvaluatedReply::payload(&r,payload.clone()).encode(&output,signed).unwrap(),old);
                }
            }
        }}
    }
    #[test]
    fn typed_pair_rejects_nonfinite_shape_and_direction() {
        let mut r=request(1,20);let mut prep=preparation(1);let mut hidden=vec![-0.;C];let base=vec![0.;C];let ax=vec![0.;R];let history=vec![-0.;3*20*256];
        assert!(direct_pair_payload(&r,&hidden,&prep,&base[..C-1],&ax,&history).is_err());
        hidden[0]=f32::NAN;assert!(direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).is_err());hidden[0]=0.1234567;assert!(direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).is_err());hidden[0]=-0.;
        prep.gates[0]=1.1;assert!(direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).is_err());prep.gates[0]=0.5;
        prep.qa[0]=f32::INFINITY;assert!(direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).is_err());prep.qa[0]=0.;
        r.op="delta_partial_mlp_prepare".into();assert!(direct_pair_payload(&r,&hidden,&prep,&base,&ax,&history).is_err());
    }
}

#[cfg(test)]
mod finish_only_tests {
    use super::*;
    #[test]fn finish_only_reply_is_separate_from_mlp_and_request_direction() {
        for n in [1,87,89] {
            let r:Request=serde_json::from_value(serde_json::json!({"version":3,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":1,"op":"delta_partial_finish","encoding":NAME,"tensor":"model.language_model.layers.29.post_attention_layernorm.weight","dims":[n,4352,4864,20,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.30.input_layernorm.weight"]})).unwrap();
            let v=vec![-0.;reply_count(&r).unwrap()];let mut b=vec![];append_reply(&mut b,&r,&v).unwrap();assert_eq!(b[0],8);
            assert!(decode_reply(&r,&b).unwrap().iter().all(|x|x.to_bits()==(-0f32).to_bits()));
            assert!(PreparedMlpDeltaStream::decode(&r,&b).is_err());
            let mut wrong=r.clone();wrong.op="delta_partial_mlp_full".into();assert!(decode_reply(&wrong,&b).is_err());
            assert!(decode_reply(&r,&b[..b.len()-1]).is_err());
            let end=b.len();b[end-2..].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(decode_reply(&r,&b).is_err());
        }
    }
}

#[cfg(test)]
mod front_follow_tests {
    use super::*;
    #[test]fn continuation_front_has_typed_stream_payload_and_checked_scope() {
        for n in [1,87,89] {let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"delta_partial_mlp_front","encoding":NAME,"tensor":"model.language_model.layers.23.post_attention_layernorm.weight","dims":[n,512,8704,12,45,6912],"scalars":[2.,1e-6],"aux":["model.language_model.layers.24.input_layernorm.weight"]})).unwrap();
            let mut v=vec![-0.;n*C];v.extend(vec![1.;n*C]);v.extend(vec![0.25;n*10]);v.extend(vec![0.1234567;n*128]);v.extend(vec![127.;n*6912]);v.extend(vec![0.0123;n*27]);v.extend(vec![0.56789;n*64]);v.extend(vec![-0.;3*20*256]);
            let mut p=vec![];append_reply(&mut p,&r,&v).unwrap();assert_eq!(p[0],9);let out=decode_reply(&r,&p).unwrap();assert!(out.iter().zip(&v).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(PreparedMlpDeltaStream::decode(&r,&p).is_err());
            assert!(decode_reply(&r,&p[..p.len()-1]).is_err());let mut bad=r.clone();bad.dims[5]=H;assert!(metadata(&bad).is_err());
            let frame=crate::encode(&r,&v).unwrap();assert!(frame.len()<2_000_000);assert!(crate::decode(&frame).is_ok());
        }
    }
}

#[cfg(test)]
mod prefix_plane_tests {
    use super::*;
    #[test]fn compressed_prefix_restores_raw_carry_and_rejects_bad_descriptors() {
        for n in [1,7,87] {
            let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"delta_partial_mlp_front","encoding":NAME,"tensor":"model.language_model.layers.23.post_attention_layernorm.weight","dims":[n,512,8704,8,45,5120],"scalars":[2.,1e-6],"aux":["model.language_model.layers.24.input_layernorm.weight"]})).unwrap();
            let prep=n*(C/256+2*R+64);let mut p=vec![10];p.extend(24u32.to_le_bytes());p.extend(vec![0;3*n*C]);
            for _ in 0..n*C/256 {p.extend(1f32.to_le_bytes());}p.extend(vec![0;4*(prep-n*C/256+n*R)]);
            for _ in 0..12 {p.push(2);p.extend(1u32.to_le_bytes());p.push(0);}
            let a=PreparedMlpDeltaStream::decode(&r,&p).unwrap();
            let mut raw=vec![2];raw.extend(vec![0;3*n*C]);for _ in 0..n*C/256 {raw.extend(1f32.to_le_bytes());}raw.extend(vec![0;4*(prep-n*C/256+n*C+n*R)]);
            raw.extend(vec![0;2*(3*24*256+45*24/2*128)+4*45*(24*128+24)]);let b=PreparedMlpDeltaStream::decode(&r,&raw).unwrap();
            let af=a.follow.as_ref().unwrap();let bf=b.follow.as_ref().unwrap();let mut aq=vec![];let mut bq=vec![];af.1.flat(n,&mut aq);bf.1.flat(n,&mut bq);assert_eq!(aq,bq);assert_eq!(af.2,bf.2);assert_eq!(af.3,bf.3);assert_eq!(a.history,b.history);assert_eq!(a.log,b.log);
            assert!(PreparedMlpDeltaStream::decode(&r,&p[..p.len()-1]).is_err());let mut bad=p.clone();bad.push(0);assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());bad=p.clone();bad[1..5].copy_from_slice(&u32::MAX.to_le_bytes());assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());bad=p.clone();let end=bad.len();bad[end-1]=255;assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());
        }
    }
}

#[cfg(test)]
mod new_tail_tests {
 use super::*;
 #[test]fn plain_mlp_rejects_bad_shape_finite_scope_and_identity() {
  let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"mlp_full_delta_partial","encoding":NAME,"tensor":"model.language_model.layers.27.post_attention_layernorm.weight","dims":[1,0,H,8,45],"scalars":[2.,1e-6],"aux":["model.language_model.layers.28.input_layernorm.weight"]})).unwrap();
  let mut p=vec![13];p.extend((4*C as u32).to_le_bytes());p.extend(vec![0;4*C+2*(3*8*256+45*4*128)+4*45*(8*128+8)]);
  assert_eq!(PreparedMlpDeltaStream::decode(&r,&p).unwrap().plain.unwrap().len(),2*C);
  for index in [0,2*C] {let mut bad=p.clone();bad[5+index..7+index].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());}
  assert!(PreparedMlpDeltaStream::decode(&r,&p[..p.len()-1]).is_err());let mut bad=r.clone();bad.dims[1]=128;bad.dims[2]=H-128;assert!(metadata(&bad).is_err());
  for op in ["delta_partial_finish","delta_partial_mlp_front","delta_partial_mlp_full"] {
   let mut follow=r.clone();follow.op=op.into();if op=="delta_partial_mlp_front"{follow.dims.push(6144);}
   assert!(metadata(&follow).is_ok());assert_eq!(inner(&follow).unwrap().op,"mlp_full_integer");
  }
  let mut invalid=r.clone();invalid.op="mlp_complete_delta_partial".into();assert!(metadata(&invalid).is_err());
  let input=PreparedMlpDeltaStream::decode(&r,&p).unwrap();bad=r.clone();bad.step+=1;
  let m=Manifest{version:1,model:r.model.clone(),pack_hash:r.pack_hash.clone(),bytes:0,tensors:vec![]};assert!(input.evaluate(&bad,&m,&mut|_,_|->Result<Vec<u8>>{panic!("identity read")}).is_err());
 }
 #[test]fn hidden_planes_restore_exactly_and_reject_unbounded_descriptors() {
  for n in [1,87,89] {
   let r:Request=serde_json::from_value(serde_json::json!({"version":1,"model":"a".repeat(64),"pack_hash":"b".repeat(64),"input_hash":"c".repeat(64),"step":7,"op":"delta_partial_mlp_front","encoding":NAME,"tensor":"model.language_model.layers.25.post_attention_layernorm.weight","dims":[n,1024,H-1024,10,45,6912],"scalars":[2.,1e-6],"aux":["model.language_model.layers.26.input_layernorm.weight"]})).unwrap();
   let prep=n*(C/256+2*R+64);let mut p=vec![12];p.extend(12u32.to_le_bytes());for _ in 0..2 {p.push(2);p.extend(1u32.to_le_bytes());p.push(0);}p.extend(24u32.to_le_bytes());p.extend(vec![0;n*C]);for _ in 0..n*C/256 {p.extend(1f32.to_le_bytes());}p.extend(vec![0;4*(prep-n*C/256+n*R)]);for _ in 0..12 {p.push(2);p.extend(1u32.to_le_bytes());p.push(0);}
   let input=PreparedMlpDeltaStream::decode(&r,&p).unwrap();assert!(input.follow.unwrap().0.iter().all(|v|v.to_bits()==0));
   let mut q=vec![14];q.extend(6u32.to_le_bytes());q.push(2);q.extend(1u32.to_le_bytes());q.push(0);q.extend_from_slice(&p[1..21]);q.extend_from_slice(&p[21+n*C..]);
   let input=PreparedMlpDeltaStream::decode(&r,&q).unwrap();assert!(input.follow.unwrap().0.iter().all(|v|v.to_bits()==0));
   for cut in [1,5,11,q.len()-1] {assert!(PreparedMlpDeltaStream::decode(&r,&q[..cut]).is_err());}
   let mut bad=q.clone();bad[1..5].copy_from_slice(&u32::MAX.to_le_bytes());assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());
   bad=q.clone();bad[10]=128;assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());bad=q.clone();bad.push(0);assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());
   for cut in [1,5,16,p.len()-1] {assert!(PreparedMlpDeltaStream::decode(&r,&p[..cut]).is_err());}let mut bad=p.clone();bad[1..5].copy_from_slice(&u32::MAX.to_le_bytes());assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());bad=p.clone();bad.push(0);assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());bad=p.clone();bad[10]=0x80;bad[16]=0x7f;assert!(PreparedMlpDeltaStream::decode(&r,&bad).is_err());
  }
 }
}
