//! Bounded diagnostic decoder for client-held exact prefix state.
use imajev_runtime::Result;
const HEAD:usize=16384;
struct Reader<'a>{bytes:&'a[u8],pos:usize}
impl<'a> Reader<'a>{
 fn take(&mut self,n:usize)->Result<&'a[u8]>{let end=self.pos.checked_add(n).ok_or("codec range")?;let b=self.bytes.get(self.pos..end).ok_or("codec truncated")?;self.pos=end;Ok(b)}
 fn u32(&mut self)->Result<u32>{Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))}
}
fn dense(table:&[u8],payload:&[u8],mant:&[u8])->Result<Vec<f32>> {
 if table.len()!=256 || mant.len()!=HEAD*3 || payload.is_empty() || payload.len()>HEAD*2 {return Err("dense shape".into());}
 let mut counts=[0u32;17];for &v in table {if v>16{return Err("Huffman length".into());}if v>0{counts[v as usize]+=1;}}
 let mut first=[0u32;17];let mut base=[0usize;17];let mut code=0u32;let mut total=0;
 for len in 1..=16 {code=(code+counts[len-1])*2;first[len]=code;base[len]=total;total+=counts[len]as usize;if code+counts[len]>1<<len {return Err("Huffman oversubscribed".into());}}
 if total==0 {return Err("empty Huffman table".into());}
 let mut symbols=Vec::with_capacity(total);for len in 1..=16 {for (s,&l) in table.iter().enumerate(){if l as usize==len{symbols.push(s as u32);}}}
 let mut fast=[0u16;1024];for len in 1..=10 {for offset in 0..counts[len] {let index=((first[len]+offset)as usize)<<(10-len);let value=((len as u16)<<8)|symbols[base[len]+offset as usize]as u16;for v in &mut fast[index..index+(1<<(10-len))]{*v=value;}}}
 let mut out=Vec::with_capacity(HEAD);let mut position=0usize;let mut reservoir=0u32;let mut bits=0usize;
 for i in 0..HEAD {
  while bits<16 && position<payload.len(){reservoir=(reservoir<<8)|payload[position]as u32;bits+=8;position+=1;}
  let key=if bits>=10 {reservoir>>(bits-10)}else{reservoir<<(10-bits)};let entry=fast[(key&1023)as usize];
  let (symbol,len)=if entry!=0 {(entry as u32&255,(entry>>8)as usize)}else{
   let mut found=None;for len in 11..=16 {if len>bits{break;}let v=reservoir>>(bits-len);if v>=first[len] && v-first[len]<counts[len] {found=Some((symbols[base[len]+(v-first[len])as usize],len));break;}}
   found.ok_or("Huffman code")?
  };
  if len>bits || symbol==255 {return Err("Huffman finite/truncated".into());}bits-=len;reservoir&=(1u32<<bits)-1;
  let low=mant[i*3]as u32|((mant[i*3+1]as u32)<<8)|((mant[i*3+2]as u32)<<16);
  out.push(f32::from_bits((low&0x7fffff)|((low>>23)<<31)|(symbol<<23)));
 }
 if position!=payload.len() || bits>7 || reservoir!=0 {return Err("Huffman trailing data".into());}Ok(out)
}
fn dense_nibble(base:u8,codes:&[u8],mant:&[u8],exceptions:&[u8],out:&mut[f32])->Result<()> {
 if base>240 || codes.len()!=HEAD/2 || mant.len()!=HEAD*3 || out.len()!=HEAD || exceptions.len()>HEAD || exceptions.contains(&255) {return Err("nibble shape/finite".into());}
 #[cfg(target_arch="wasm32")]
 // All fixed-size source/destination spans are established above. The helper
 // checks every exceptional exponent before reading it and writes all lanes.
 return unsafe {nibble_simd(base,codes,mant,exceptions,out)};
 #[cfg(not(target_arch="wasm32"))]
 {let mut e=0;for i in 0..HEAD {let code=(codes[i/2]>>((i%2)*4))&15;let exponent=if code==15 {let v=*exceptions.get(e).ok_or("nibble exception")?;e+=1;v}else{base+code};let low=mant[i*3]as u32|((mant[i*3+1]as u32)<<8)|((mant[i*3+2]as u32)<<16);out[i]=f32::from_bits((low&0x7fffff)|((low>>23)<<31)|((exponent as u32)<<23));}if e!=exceptions.len(){return Err("nibble trailing exceptions".into());}Ok(())}
}
#[cfg(target_arch="wasm32")]
#[target_feature(enable="simd128")]
unsafe fn nibble_simd(base:u8,codes:&[u8],mant:&[u8],exceptions:&[u8],out:&mut[f32])->Result<()> {
 use core::arch::wasm32::*;
 let mut e=0;let mask=i32x4_splat(0x7fffff);let sign=i32x4_splat(0x800000);
 for i in (0..HEAD).step_by(4) {
  let packed=u16::from_le_bytes([*codes.as_ptr().add(i/2),*codes.as_ptr().add(i/2+1)]);
  let mut c=[(packed&15)as u8,((packed>>4)&15)as u8,((packed>>8)&15)as u8,((packed>>12)&15)as u8];
  let mut exponent=i32x4(c[0]as i32,c[1]as i32,c[2]as i32,c[3]as i32);
  if v128_any_true(i32x4_eq(exponent,i32x4_splat(15))) {
   for v in &mut c {if *v==15 {*v=*exceptions.get(e).ok_or("nibble exception")?;e+=1;}else{*v+=base;}}
   exponent=i32x4(c[0]as i32,c[1]as i32,c[2]as i32,c[3]as i32);
  }else{exponent=i32x4_add(exponent,i32x4_splat(base as i32));}
  // Exactly 12 mantissa bytes are available even for the final four floats.
  let p=mant.as_ptr().add(i.wrapping_mul(3));let a=v128_load64_zero(p.cast());let b=v128_load32_zero(p.add(8).cast());
  let low=i8x16_shuffle::<0,1,2,20,3,4,5,20,6,7,16,20,17,18,19,20>(a,b);
  let value=v128_or(v128_or(v128_and(low,mask),i32x4_shl(v128_and(low,sign),8)),i32x4_shl(exponent,23));
  v128_store(out.as_mut_ptr().add(i).cast(),value);
 }
 if e!=exceptions.len(){return Err("nibble trailing exceptions".into());}Ok(())
}
pub fn decode(bytes:&[u8])->Result<Vec<f32>> {
 if bytes.len()>1_990_000 {return Err("hybrid size".into());}
 let mut r=Reader{bytes,pos:0};let magic=r.take(4)?;let nibble=magic==b"NPF1";if !nibble && magic!=b"HPF1" {return Err("hybrid magic".into());}let n=r.u32()?as usize;let mask=r.u32()?;
 if n==0 || n>132 || (0..16).any(|j|((mask>>(j*2))&3)!=0 && ((mask>>(j*2))&3)!=3) {return Err("hybrid mask/tokens".into());}
 let mut result=vec![0.;32*HEAD];
 for h in 0..32 {if mask&(1<<h)!=0 {if nibble {let metadata=r.take(8)?;if metadata[1..4]!=[0,0,0] {return Err("nibble reserved".into());}let count=u32::from_le_bytes(metadata[4..8].try_into().unwrap())as usize;if count>HEAD{return Err("nibble exception count".into());}let codes=r.take(HEAD/2)?;let mant=r.take(HEAD*3)?;let exceptions=r.take(count)?;dense_nibble(metadata[0],codes,mant,exceptions,&mut result[h*HEAD..(h+1)*HEAD])?;continue;}let table=r.take(256)?;let count=r.u32()?as usize;let mant_count=r.u32()?as usize;if count>HEAD*2 || mant_count!=HEAD*3 {return Err("dense lengths".into());}let payload=r.take(count)?;let mant=r.take(mant_count)?;let state=dense(table,payload,mant)?;result[h*HEAD..(h+1)*HEAD].copy_from_slice(&state);}}
 let heads:Vec<_>=(0..32).filter(|h|mask&(1<<h)==0).collect();let h=heads.len();
 if h>0 {let kc=n*h/2*128;let vc=n*h*128;let mut log=Vec::with_capacity(kc+vc+n*h);
  for b in r.take(kc*2)?.chunks_exact(2){let v=f32::from_bits((u16::from_le_bytes(b.try_into().unwrap())as u32)<<16);if !v.is_finite(){return Err("hybrid K finite".into());}log.push(v);}
  for b in r.take((vc+n*h)*4)?.chunks_exact(4){log.push(f32::from_le_bytes(b.try_into().unwrap()));}
  let state=imajev_runtime::delta_log::restore(n,h,&log)?;
  for (i,head) in heads.into_iter().enumerate(){result[head*HEAD..(head+1)*HEAD].copy_from_slice(&state[i*HEAD..(i+1)*HEAD]);}
 }
 if r.pos!=bytes.len(){return Err("hybrid trailing bytes".into());}Ok(result)
}
/// Build a client-held exact prefix packet once, from state produced on-canister.
/// `state` must come from the same log; the query wrapper constructs it directly.
fn encode_state(log:&[f32],state:&[f32],n:usize)->Result<Vec<u8>> {
 if n==0 || n>132 || log.len()!=n*6176 || state.len()!=32*HEAD || !log.iter().chain(state).all(|v|v.is_finite()) || log[..n*2048].iter().any(|v|v.to_bits()&65535!=0) {return Err("prefix encode shape/finite/BF16 K".into());}
 let mut bodies=Vec::with_capacity(32);
 for head in state.chunks_exact(HEAD) {
  let mut histogram=[0usize;256];for v in head {histogram[((v.to_bits()>>23)&255)as usize]+=1;}
  let mut inside:usize=histogram[..15].iter().sum();let mut best=inside;let mut base=0usize;
  for b in 1..=240 {inside=inside-histogram[b-1]+histogram[b+14];if inside>best {best=inside;base=b;}}
  let mut codes=Vec::with_capacity(HEAD/2);let mut mant=Vec::with_capacity(HEAD*3);let mut exceptions=Vec::with_capacity(HEAD-best);
  for (i,v) in head.iter().enumerate() {
   let bits=v.to_bits();let exp=((bits>>23)&255)as usize;
   let code=if (base..base+15).contains(&exp) {(exp-base)as u8}else{exceptions.push(exp as u8);15};
   if i%2==0 {codes.push(code);}else{*codes.last_mut().unwrap()|=code<<4;}
   let low=(bits&0x7fffff)|((bits>>31)<<23);mant.extend_from_slice(&low.to_le_bytes()[..3]);
  }
  let mut body=Vec::with_capacity(8+codes.len()+mant.len()+exceptions.len());body.extend([base as u8,0,0,0]);body.extend((exceptions.len()as u32).to_le_bytes());body.extend(codes);body.extend(mant);body.extend(exceptions);bodies.push(body);
 }
 // Select nine complete shared-K pairs. Full-model frame sizing is still a
 // separate integration check; this diagnostic never changes its 2 MB bound.
 let mut pairs:Vec<_>=(0..16).collect();pairs.sort_by_key(|&p|(bodies[2*p].len()+bodies[2*p+1].len(),p));
 let mask=pairs[..9].iter().fold(0u32,|mask,&p|mask|(3u32<<(2*p)));
 let mut packet=b"NPF1".to_vec();packet.extend((n as u32).to_le_bytes());packet.extend(mask.to_le_bytes());
 for (h,body) in bodies.into_iter().enumerate(){if mask&(1<<h)!=0{packet.extend(body);}}
 let remaining:Vec<_>=(0..32).filter(|h|mask&(1<<h)==0).collect();
 for t in 0..n {for pair in 0..16 {if mask&(3u32<<(2*pair))==0 {for v in &log[t*2048+pair*128..t*2048+(pair+1)*128] {packet.extend(((v.to_bits()>>16)as u16).to_le_bytes());}}}}
 for t in 0..n {for &h in &remaining {for v in &log[n*2048+t*4096+h*128..n*2048+t*4096+(h+1)*128] {packet.extend(v.to_le_bytes());}}}
 for t in 0..n {for &h in &remaining {packet.extend(log[n*6144+t*32+h].to_le_bytes());}}
 if packet.len()>1_990_000 {return Err("prefix encoded size".into());}Ok(packet)
}
/// No question state is retained: the returned packet belongs to the client.
pub fn prepare(log:&[f32],n:usize)->Result<(Vec<u8>,Vec<f32>)> {
 if n==0 || n>132 || log.len()!=n*6176 {return Err("prefix prepare shape".into());}
 let state=imajev_runtime::delta_log::restore(n,32,log)?;
 let packet=encode_state(log,&state,n)?;Ok((packet,state))
}

#[cfg(test)]mod tests {use super::*;
 #[test]fn prepare_round_trip_and_invalid_input(){
  let n=2;let mut log=vec![0f32;n*6176];for v in &mut log[..n*2048]{*v=f32::from_bits(0x3e800000);}
  for (i,v) in log[n*2048..n*6144].iter_mut().enumerate(){*v=(i%127)as f32*0.0123;}
  log[n*6144..].fill(0.75);let(packet,state)=prepare(&log,n).unwrap();let out=decode(&packet).unwrap();assert!(state.iter().zip(out).all(|(a,b)|a.to_bits()==b.to_bits()));
  assert!(prepare(&[],0).is_err());assert!(prepare(&log,133).is_err());log[0]=0.250001;assert!(prepare(&log,n).is_err());log[0]=f32::INFINITY;assert!(prepare(&log,n).is_err());
 }
 fn one_head()->Vec<u8>{let mut b=vec![0;256];b[127]=1;b.extend(((HEAD/8)as u32).to_le_bytes());b.extend((HEAD as u32*3).to_le_bytes());b.extend(vec![0;HEAD/8]);b.extend(vec![0;HEAD*3]);b}
 #[test]fn all_dense_exact_and_malformed(){let mut b=b"HPF1".to_vec();b.extend(45u32.to_le_bytes());b.extend(u32::MAX.to_le_bytes());for _ in 0..32 {b.extend(one_head());}let out=decode(&b).unwrap();assert!(out.iter().all(|v|v.to_bits()==1f32.to_bits()));let mut trailing=b.clone();trailing.push(0);assert!(decode(&trailing).is_err());for n in [0,3,12,b.len()-1] {assert!(decode(&b[..n]).is_err());}let mut invalid=b.clone();invalid[8..12].copy_from_slice(&1u32.to_le_bytes());assert!(decode(&invalid).is_err());invalid=b.clone();invalid[12]=17;assert!(decode(&invalid).is_err());}
 #[test]fn nibble_extremes_and_rejection(){let mut codes=Vec::new();let mut mant=Vec::new();let mut exceptions=Vec::new();let mut expected=Vec::new();
  for i in 0..HEAD {let code=[0u8,15,14,1][i%4];if i%2==0{codes.push(code);}else{*codes.last_mut().unwrap()|=code<<4;}let low=[0u32,0x800000,0x7fffff,0xffffff][i%4];mant.extend_from_slice(&low.to_le_bytes()[..3]);let exp=if code==15 {let v=if i/4%2==0{0}else{254};exceptions.push(v);v}else{100+code};expected.push((low&0x7fffff)|((low>>23)<<31)|((exp as u32)<<23));}
  let mut body=vec![100,0,0,0];body.extend((exceptions.len()as u32).to_le_bytes());body.extend(&codes);body.extend(&mant);body.extend(&exceptions);
  let mut packet=b"NPF1".to_vec();packet.extend(45u32.to_le_bytes());packet.extend(u32::MAX.to_le_bytes());for _ in 0..32 {packet.extend(&body);}
  let out=decode(&packet).unwrap();for head in out.chunks_exact(HEAD){assert!(head.iter().zip(&expected).all(|(a,b)|a.to_bits()==*b));}
  let mut wrong=packet.clone();wrong[13]=1;assert!(decode(&wrong).is_err());let mut out=vec![0.;HEAD];assert!(dense_nibble(241,&codes,&mant,&exceptions,&mut out).is_err());assert!(dense_nibble(100,&codes,&mant,&exceptions[..1],&mut out).is_err());let mut bad=exceptions.clone();bad[0]=255;assert!(dense_nibble(100,&codes,&mant,&bad,&mut out).is_err());
 }

}
