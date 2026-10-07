//! Decode saved Candid capture chunks without relying on displayed formatting.
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=std::env::args().collect();let text=std::fs::read_to_string(&a[1])?;
 let h=text.trim().trim_start_matches("0x");
 let raw=(0..h.len()).step_by(2).map(|i|u8::from_str_radix(&h[i..i+2],16)).collect::<Result<Vec<_>,_>>()?;
 let bytes=candid::decode_one::<Result<Vec<u8>,String>>(&raw)??;
 std::fs::write(&a[2],bytes)?;Ok(())
}
