//! Bounded lossless byte planes. No dictionary or per-byte LZ state.
use crate::Result;
fn huffman(table:&[u8],payload:&[u8],out:&mut[u8])->Result<()> {
 let mut counts=[0u32;25];for &len in table {if len>24{return Err("carry Huffman length".into());}if len>0{counts[len as usize]+=1;}}
 let(mut first,mut base)=([0u32;25],[0usize;25]);let(mut code,mut total)=(0u32,0usize);
 for len in 1..=24 {code=(code+counts[len-1])*2;first[len]=code;base[len]=total;total+=counts[len]as usize;if code+counts[len]>1<<len{return Err("carry Huffman oversubscribed".into());}}
 if total<2 || code+counts[24]!=1<<24{return Err("carry Huffman incomplete".into());}
 let mut symbols=[0u8;256];let mut pos=0;for len in 1..=24 {for(s,&l)in table.iter().enumerate(){if l as usize==len{symbols[pos]=s as u8;pos+=1;}}}
 let mut fast=[0u16;1024];for len in 1..=10 {for offset in 0..counts[len]{let index=((first[len]+offset)as usize)<<(10-len);let value=((len as u16)<<8)|symbols[base[len]+offset as usize]as u16;fast[index..index+(1<<(10-len))].fill(value);}}
 let(mut position,mut reservoir,mut bits)=(0usize,0u32,0usize);
 for v in out {
  while bits<24 && position<payload.len(){reservoir=(reservoir<<8)|payload[position]as u32;bits+=8;position+=1;}
  let key=if bits>=10{reservoir>>(bits-10)}else{reservoir<<(10-bits)};let entry=fast[(key&1023)as usize];
  let(symbol,len)=if entry!=0{(entry as u8,(entry>>8)as usize)}else{
   let mut found=None;for len in 11..=24 {if len>bits{break;}let v=reservoir>>(bits-len);if v>=first[len] && v-first[len]<counts[len]{found=Some((symbols[base[len]+(v-first[len])as usize],len));break;}}found.ok_or("carry Huffman code")?
  };
  if len>bits{return Err("carry Huffman truncated".into());}bits-=len;reservoir&=(1u32<<bits)-1;*v=symbol;
 }
 if position!=payload.len() || bits>7 || reservoir!=0{return Err("carry Huffman trailing/padding".into());}Ok(())
}
pub(crate) fn decode(bytes:&[u8],shapes:&[(usize,usize)])->Result<Vec<Vec<u8>>> {
 let mut cursor=0usize;let mut result=Vec::with_capacity(shapes.len());
 for &(count,width)in shapes {let mut group=vec![0;count*width];for plane in 0..width {
  let desc=bytes.get(cursor..cursor+5).ok_or("carry plane descriptor")?;cursor+=5;let mode=desc[0];let len=u32::from_le_bytes(desc[1..].try_into().unwrap())as usize;
  // No allocation derives from the compressed descriptor.
  let mut lane=vec![0;count];
  if mode==1 {let table=bytes.get(cursor..cursor+256).ok_or("carry plane table")?;cursor+=256;let end=cursor.checked_add(len).ok_or("carry plane range")?;let payload=bytes.get(cursor..end).ok_or("carry plane truncated")?;huffman(table,payload,&mut lane)?;cursor=end;}
  else {let end=cursor.checked_add(len).ok_or("carry plane range")?;let payload=bytes.get(cursor..end).ok_or("carry plane truncated")?;cursor=end;match mode {0 if len==count=>lane.copy_from_slice(payload),2 if len==1 && count>0=>lane.fill(payload[0]),_=>return Err("carry plane mode/length".into())}}
  group[plane*count..(plane+1)*count].copy_from_slice(&lane);
 }result.push(group);}
 if cursor!=bytes.len(){return Err("carry plane trailing".into());}Ok(result)
}
#[cfg(test)]mod tests {use super::*;
 #[test]fn exact_and_malformed(){let mut table=[0u8;256];table[0]=1;table[255]=1;let mut b=vec![1];b.extend(1u32.to_le_bytes());b.extend(table);b.push(0x50);assert_eq!(decode(&b,&[(4,1)]).unwrap()[0],[0,255,0,255]);for cut in 0..b.len(){assert!(decode(&b[..cut],&[(4,1)]).is_err());}let mut bad=b.clone();*bad.last_mut().unwrap()|=1;assert!(decode(&bad,&[(4,1)]).is_err());bad=b.clone();bad.push(0);assert!(decode(&bad,&[(4,1)]).is_err());bad=b.clone();bad[5]=2;assert!(decode(&bad,&[(4,1)]).is_err());}
 #[test]fn raw_constant_empty(){let mut b=vec![0];b.extend(3u32.to_le_bytes());b.extend([1,2,3]);b.push(2);b.extend(1u32.to_le_bytes());b.push(255);assert_eq!(decode(&b,&[(3,2)]).unwrap()[0],[1,2,3,255,255,255]);let mut empty=vec![0];empty.extend(0u32.to_le_bytes());assert!(decode(&empty,&[(0,1)]).is_ok());}
}
