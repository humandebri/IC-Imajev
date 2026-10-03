//! Bounded lossless byte planes. No dictionary or per-byte LZ state.
use crate::Result;
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
  else {let end=cursor.checked_add(len).ok_or("carry plane range")?;let payload=bytes.get(cursor..end).ok_or("carry plane truncated")?;cursor=end;match mode {0 if len==count=>lane.copy_from_slice(payload),2 if len==1 && count>0=>lane.fill(payload[0]),_=>return Err("carry plane mode/length".into())}}

 }result.push(group);}
 if cursor!=bytes.len(){return Err("carry plane trailing".into());}Ok(result)
}
#[cfg(test)]mod tests {use super::*;
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
