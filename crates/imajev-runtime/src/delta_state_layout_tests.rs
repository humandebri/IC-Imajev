use crate::*;
#[test]
fn key_major_recurrence_matches_output_and_final_state() {
 for n in [1,7,8,32,45,80,87,89,132] {for (dk,dv) in [(3,4),(128,12),(128,16),(128,128)] {
  let q:Vec<_>=(0..n*dk).map(|i|((i%17)as f32-8.)/128.).collect();
  let k:Vec<_>=(0..n*dk).map(|i|((i%13)as f32-6.)/128.).collect();
  let v:Vec<_>=(0..n*dv).map(|i|((i%19)as f32-9.)/32.).collect();
  let g:Vec<_>=(0..n).map(|i|[0.,1.,0.875][i%3]).collect();
  let b:Vec<_>=(0..n).map(|i|[1.,0.,0.5][i%3]).collect();
  let mut normal:Vec<_>=(0..dk*dv).map(|i|if i%5==0{-0.}else{((i%23)as f32-11.)/64.}).collect();
  let mut key=vec![0.;dk*dv];for d in 0..dv {for i in 0..dk {key[i*dv+d]=normal[d*dk+i];}}
  let expected=delta(&q,&k,&v,&g,&b,&mut normal,dk,dv).unwrap();
  let actual=delta_from_key_major(&q,&k,&v,&g,&b,&mut key,dk,dv).unwrap();
  assert!(actual.iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));
  for d in 0..dv {for i in 0..dk {assert_eq!(key[i*dv+d].to_bits(),normal[d*dk+i].to_bits());}}
 }}
}
#[test]
fn key_major_log_restore_matches_existing_states() {
 for n in [1,7,45] {for h in [2,14,32] {
  let mut log:Vec<_>=(0..n*h/2*128).map(|i|bf(((i%17)as f32-8.)/64.)).collect();
  log.extend((0..n*h*128).map(|i|((i%13)as f32-6.)/32.));
  log.extend((0..n*h).map(|i|[0.,1.,0.875][i%3]));
  let expected=delta_log::restore(n,h,&log).unwrap();let key=delta_log::restore_key_major(n,h,&log).unwrap();
  for head in 0..h {for d in 0..128 {for i in 0..128 {assert_eq!(expected[head*16384+d*128+i].to_bits(),key[head*16384+i*128+d].to_bits());}}}
 }}
}
#[test]
fn typed_packet_decoder_preserves_state_and_rejects_invalid_packet() {
 for n in [1,7,45] {
  let mut log:Vec<_>=(0..n*2048).map(|i|bf(((i%17)as f32-8.)/32.)).collect();
  log.extend((0..n*4096).map(|i|((i%29)as f32-14.)/64.));
  log.extend((0..n*32).map(|i|[0.,1.,0.875][i%3]));
  let(packet,expected)=prefix_hybrid_codec::prepare(&log,n).unwrap();
  let typed=prefix_hybrid_codec::decode_for_delta(&packet).unwrap();
  assert!(matches!(typed,delta_full_log::InitialState::KeyMajor(_)));
  assert!(typed.value_major().iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));
  let mut invalid=packet.clone();invalid[8..12].copy_from_slice(&1u32.to_le_bytes());assert!(prefix_hybrid_codec::decode_for_delta(&invalid).is_err());
  assert!(prefix_hybrid_codec::decode_for_delta(&packet[..packet.len()-1]).is_err());
 }
}
#[test]
fn in_place_transpose_preserves_all_bits_and_is_involution() {
 let original:Vec<_>=(0..16384).map(|i|f32::from_bits((i as u32).wrapping_mul(2654435761))).collect();
 let mut head=original.clone();prefix_hybrid_codec::transpose_head(&mut head);
 for row in 0..128 {for col in 0..128 {assert_eq!(head[row*128+col].to_bits(),original[col*128+row].to_bits());}}
 prefix_hybrid_codec::transpose_head(&mut head);assert!(head.iter().zip(original).all(|(a,b)|a.to_bits()==b.to_bits()));
}
