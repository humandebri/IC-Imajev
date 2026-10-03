use imajev_runtime::{encode, execute, Manifest};
#[cfg(not(feature="experimental-projection-reuse"))]
use imajev_runtime::{decode, evaluate_with_reader};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let frame=std::fs::read(&a[1])?;
    #[cfg(feature="experimental-projection-reuse")]
    let (mut r,input)=imajev_runtime::decode_query(&frame)?;
    #[cfg(feature="experimental-projection-reuse")]
    let x=input.values().unwrap_or(&[]);
    #[cfg(not(feature="experimental-projection-reuse"))]
    let (mut r,x)=decode(&frame)?;
    let y = if a.len() >= 5 {
        let m: Manifest = serde_json::from_slice(&std::fs::read(&a[3])?)?;
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&a[4])?;
        let read = |offset, len| {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            let mut bytes = vec![0; len];
            file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            Ok(bytes)
        };
        #[cfg(feature="experimental-projection-reuse")]
        let result=imajev_runtime::evaluate_owned_decoded_with_prepared_buffer(&r,input,&m,read)?;
        #[cfg(not(feature="experimental-projection-reuse"))]
        let result=evaluate_with_reader(&r,&x,&m,read)?;
        result.0
    } else {
        execute(&r, &x, &[])?
    };
    r.step = r.step.checked_add(1).ok_or("progress overflow")?;
    std::fs::write(&a[2], encode(&r, &y)?)?;
    Ok(())
}
