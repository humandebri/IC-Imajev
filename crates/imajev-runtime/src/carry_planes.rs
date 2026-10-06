//! Bounded lossless byte planes. No per-byte LZ state.
use crate::Result;
pub(crate) fn interleave(planes:&[u8],out:&mut[u8])->Result<()> {
 if planes.len()!=out.len() || planes.len()%2!=0{return Err("carry interleave shape".into());}
 let count=out.len()/2;let mut begin=0;
 #[cfg(target_arch="wasm32")]
 unsafe {
  use core::arch::wasm32::*;
  // planes consists of two count-byte lanes; every vector iteration reads
  // sixteen bytes in each lane and writes thirty-two disjoint output bytes.
  while begin+16<=count {
   let low=v128_load(planes.as_ptr().add(begin).cast());let high=v128_load(planes.as_ptr().add(count+begin).cast());
   v128_store(out.as_mut_ptr().add(begin*2).cast(),i8x16_shuffle::<0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23>(low,high));
   v128_store(out.as_mut_ptr().add(begin*2+16).cast(),i8x16_shuffle::<8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31>(low,high));
   begin+=16;
  }
 }
 for i in begin..count {out[2*i]=planes[i];out[2*i+1]=planes[count+i];}Ok(())
}
fn dictionary(payload:&[u8],out:&mut[u8])->Result<()> {
 let &count=payload.first().ok_or("carry dictionary header")?;let count=count as usize;
 if count==0 || count>15 || out.is_empty(){return Err("carry dictionary bounds".into());}
 let table=payload.get(1..1+count).ok_or("carry dictionary table")?;
 let mut seen=[false;256];for &v in table {if seen[v as usize]{return Err("carry dictionary duplicate".into());}seen[v as usize]=true;}
 let packed=(out.len()+1)/2;let codes=payload.get(1+count..1+count+packed).ok_or("carry dictionary codes")?;
 let escape=&payload[1+count+packed..];let mut pos=0;
 if out.len()%2==1 && codes[packed-1]>>4!=0{return Err("carry dictionary padding".into());}
 let mut start=0;
 #[cfg(target_arch="wasm32")]
 {
  use core::arch::wasm32::*;
  let mut padded=[0u8;16];padded[..count].copy_from_slice(table);
  // Codes has ceil(out.len()/2) bytes. Each iteration reads eight code
  // bytes and writes sixteen disjoint output bytes; padded has sixteen.
  // Invalid dictionary indices are rejected before the swizzle/store.
  unsafe {
   let dict=v128_load(padded.as_ptr().cast());let mask=i8x16_splat(15);let limit=i8x16_splat(count as i8-1);
   let odd=i8x16(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1);
   while start+16<=out.len(){
    let packed=v128_load64_zero(codes.as_ptr().add(start/2).cast());
    let doubled=i8x16_shuffle::<0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7>(packed,packed);
    let indices=v128_and(v128_bitselect(u8x16_shr(doubled,4),doubled,odd),mask);
    let escaped=i8x16_eq(indices,mask);
    if i8x16_bitmask(v128_andnot(u8x16_gt(indices,limit),escaped))!=0{return Err("carry dictionary index".into());}
    if i8x16_bitmask(escaped)==0{v128_store(out.as_mut_ptr().add(start).cast(),i8x16_swizzle(dict,indices));}
    else {for i in start..start+16 {let code=(codes[i/2]>>(4*(i%2)))as usize&15;out[i]=if code==15{let v=*escape.get(pos).ok_or("carry dictionary escape")?;pos+=1;v}else{table[code]};}}
    start+=16;
   }
  }
 }
 for i in start..out.len(){let v=&mut out[i];let code=(codes[i/2]>>(4*(i%2)))as usize&15;
  if code==15 {*v=*escape.get(pos).ok_or("carry dictionary escape")?;pos+=1;}
  else {*v=*table.get(code).ok_or("carry dictionary index")?;}
 }
 if pos!=escape.len(){return Err("carry dictionary trailing".into());}Ok(())
}
fn huffman(table:&[u8],payload:&[u8],out:&mut[u8])->Result<()> {
 let mut counts=[0u32;25];for &len in table {if len>24{return Err("carry Huffman length".into());}if len>0{counts[len as usize]+=1;}}
 let(mut first,mut base)=([0u32;25],[0usize;25]);let(mut code,mut total)=(0u32,0usize);
 for len in 1..=24 {code=(code+counts[len-1])*2;first[len]=code;base[len]=total;total+=counts[len]as usize;if code+counts[len]>1<<len{return Err("carry Huffman oversubscribed".into());}}
 if total<2 || code+counts[24]!=1<<24{return Err("carry Huffman incomplete".into());}
 let mut symbols=[0u8;256];let mut pos=0;for len in 1..=24 {for(s,&l)in table.iter().enumerate(){if l as usize==len{symbols[pos]=s as u8;pos+=1;}}}
 let mut fast=[0u16;1024];for len in 1..=10 {for offset in 0..counts[len]{let index=((first[len]+offset)as usize)<<(10-len);let value=((len as u16)<<8)|symbols[base[len]+offset as usize]as u16;fast[index..index+(1<<(10-len))].fill(value);}}
 let mut pair=vec![0u32;16384];
 let mut words=Vec::with_capacity(total);
 for len in 1..=14 {for offset in 0..counts[len] {words.push((first[len]+offset,len,symbols[base[len]+offset as usize]));}}
 for &(a,la,sa)in &words {if la+words[0].1>14{break;}for &(b,lb,sb)in &words {let len=la+lb;if len>14{break;}if len<=14 {let prefix=((a<<lb)|b)as usize;let start=prefix<<(14-len);let entry=((len as u32)<<16)|(sa as u32)|((sb as u32)<<8);pair[start..start+(1<<(14-len))].fill(entry);}}}
 let(mut position,mut reservoir,mut bits,mut index)=(0usize,0u64,0usize,0usize);
 while index<out.len() {
  if bits<24 && position+4<=payload.len(){reservoir=(reservoir<<32)|u32::from_be_bytes(payload[position..position+4].try_into().unwrap())as u64;bits+=32;position+=4;}
  while bits<24 && position<payload.len(){reservoir=(reservoir<<8)|payload[position]as u64;bits+=8;position+=1;}
  if bits>=14 && index+1<out.len() {
   let entry=pair[((reservoir>>(bits-14))&16383)as usize];
   if entry!=0 {let len=(entry>>16)as usize;bits-=len;reservoir&=(1u64<<bits)-1;out[index]=entry as u8;out[index+1]=(entry>>8)as u8;index+=2;continue;}
  }
  let key=if bits>=10{reservoir>>(bits-10)}else{reservoir<<(10-bits)};let entry=fast[(key&1023)as usize];
  let(symbol,len)=if entry!=0{(entry as u8,(entry>>8)as usize)}else{
   let mut found=None;for len in 11..=24 {if len>bits{break;}let v=(reservoir>>(bits-len))as u32;if v>=first[len] && v-first[len]<counts[len]{found=Some((symbols[base[len]+(v-first[len])as usize],len));break;}}found.ok_or("carry Huffman code")?
  };
  if len>bits{return Err("carry Huffman truncated".into());}bits-=len;reservoir&=(1u64<<bits)-1;out[index]=symbol;index+=1;
 }
 if position!=payload.len() || bits>7 || reservoir!=0{return Err("carry Huffman trailing/padding".into());}Ok(())
}
pub(crate) fn decode(bytes:&[u8],shapes:&[(usize,usize)])->Result<Vec<Vec<u8>>> {
 let mut cursor=0usize;let mut result=Vec::with_capacity(shapes.len());
 for &(count,width)in shapes {let mut group=vec![0;count*width];for plane in 0..width {
  let desc=bytes.get(cursor..cursor+5).ok_or("carry plane descriptor")?;cursor+=5;let mode=desc[0];let len=u32::from_le_bytes(desc[1..].try_into().unwrap())as usize;
  // No allocation derives from the compressed descriptor.
  let lane=&mut group[plane*count..(plane+1)*count];
  if mode==1 {let table=bytes.get(cursor..cursor+256).ok_or("carry plane table")?;cursor+=256;let end=cursor.checked_add(len).ok_or("carry plane range")?;let payload=bytes.get(cursor..end).ok_or("carry plane truncated")?;huffman(table,payload,lane)?;cursor=end;}
  else {let end=cursor.checked_add(len).ok_or("carry plane range")?;let payload=bytes.get(cursor..end).ok_or("carry plane truncated")?;cursor=end;match mode {0 if len==count=>lane.copy_from_slice(payload),2 if len==1 && count>0=>lane.fill(payload[0]),3=>dictionary(payload,lane)?,_=>return Err("carry plane mode/length".into())}}

 }result.push(group);}
 if cursor!=bytes.len(){return Err("carry plane trailing".into());}Ok(result)
}
#[cfg(test)]mod tests {use super::*;
 #[test]fn interleave_shapes_and_tail(){for count in [0,1,7,15,16,17,31,32,33,2560]{let p:Vec<_>=(0..count*2).map(|i|i as u8).collect();let mut out=vec![0;p.len()];interleave(&p,&mut out).unwrap();for i in 0..count{assert_eq!(&out[2*i..2*i+2],&[p[i],p[count+i]]);}}assert!(interleave(&[0],&mut[0]).is_err());assert!(interleave(&[0,1],&mut[]).is_err());}
 #[test]fn dictionary_exact_and_rejected(){
  let payload=[2,0,128,0x10,0x1f,255];let mut out=[0u8;4];dictionary(&payload,&mut out).unwrap();assert_eq!(out,[0,128,255,128]);
  for cut in 0..payload.len(){assert!(dictionary(&payload[..cut],&mut out).is_err());}
  for p in [vec![0],vec![16],vec![2,0,0,0x10,0x1f,255],vec![2,0,128,0x20,0x1f,255],vec![2,0,128,0x10,0x1f,255,0]] {assert!(dictionary(&p,&mut out).is_err());}
  assert!(dictionary(&[1,128,0x10],&mut[0u8;1]).is_err());
  let mut wire=vec![3];wire.extend((payload.len()as u32).to_le_bytes());wire.extend(payload);assert_eq!(decode(&wire,&[(4,1)]).unwrap()[0],out);
 }
 #[test]fn exact_and_malformed(){let mut table=[0u8;256];table[0]=1;table[255]=1;let mut b=vec![1];b.extend(1u32.to_le_bytes());b.extend(table);b.push(0x50);assert_eq!(decode(&b,&[(4,1)]).unwrap()[0],[0,255,0,255]);for cut in 0..b.len(){assert!(decode(&b[..cut],&[(4,1)]).is_err());}let mut bad=b.clone();*bad.last_mut().unwrap()|=1;assert!(decode(&bad,&[(4,1)]).is_err());bad=b.clone();bad.push(0);assert!(decode(&bad,&[(4,1)]).is_err());bad=b.clone();bad[5]=2;assert!(decode(&bad,&[(4,1)]).is_err());}
 #[test]fn raw_constant_empty(){let mut b=vec![0];b.extend(3u32.to_le_bytes());b.extend([1,2,3]);b.push(2);b.extend(1u32.to_le_bytes());b.push(255);assert_eq!(decode(&b,&[(3,2)]).unwrap()[0],[1,2,3,255,255,255]);let mut empty=vec![0];empty.extend(0u32.to_le_bytes());assert!(decode(&empty,&[(0,1)]).is_ok());}
 #[test]fn paired_word_decoder_handles_odd_tails_and_24_bit_codes(){
  let mut table=[0u8;256];for i in 0..23{table[i]=i as u8+1;}table[23]=24;table[24]=24;
  let mut codes=[0u32;25];let mut code=0u32;let mut previous=0;for i in 0..25{let len=table[i]as usize;code<<=len-previous;codes[i]=code;code+=1;previous=len;}
  for n in [1,2,3,7,8,9,31,32,33,257] {
   let raw:Vec<_>=(0..n).map(|i|(i%25)as u8).collect();let(mut bits,mut reservoir)=(0usize,0u64);let mut payload=vec![];
   for &symbol in &raw {let len=table[symbol as usize]as usize;reservoir=(reservoir<<len)|codes[symbol as usize]as u64;bits+=len;while bits>=8{bits-=8;payload.push((reservoir>>bits)as u8);}reservoir&=(1u64<<bits)-1;}
   if bits>0{payload.push((reservoir<<(8-bits))as u8);}let mut out=vec![0;n];huffman(&table,&payload,&mut out).unwrap();assert_eq!(out,raw);let mut trailing=payload.clone();trailing.push(0);assert!(huffman(&table,&trailing,&mut out).is_err());assert!(huffman(&table,&payload[..payload.len()-1],&mut out).is_err());
  }
 }

}
