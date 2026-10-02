use imajev_runtime::{decode, encode, evaluate_with_reader, execute, Manifest};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let (mut r, x) = decode(&std::fs::read(&a[1])?)?;
    let y = if a.len() >= 5 {
        let m: Manifest = serde_json::from_slice(&std::fs::read(&a[3])?)?;
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&a[4])?;
        evaluate_with_reader(&r, &x, &m, |offset, len| {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            let mut bytes = vec![0; len];
            file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            Ok(bytes)
        })?
        .0
    } else {
        execute(&r, &x, &[])?
    };
    r.step = r.step.checked_add(1).ok_or("progress overflow")?;
    std::fs::write(&a[2], encode(&r, &y)?)?;
    Ok(())
}
