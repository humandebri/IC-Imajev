// Refuse replacing a kernel when the compiler merged its export with another.
use std::{env,fs,path::Path};
use serde::Deserialize;
type Error = Box<dyn std::error::Error>;
fn sha(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
use wasmparser::{Parser,Payload,TypeRef,ValType,Validator,WasmFeatures};
use sha2::{Digest,Sha256};
fn leb(mut v:usize)->Vec<u8>{let mut b=vec![];loop{let x=(v&127)as u8;v>>=7;b.push(x|if v!=0{128}else{0});if v==0{return b;}}}
fn take_leb(b:&[u8],p:&mut usize)->usize{let mut x=0usize;let mut shift=0;loop{let v=b[*p];*p+=1;x|=((v&127)as usize)<<shift;if v&128==0{return x;}shift+=7;assert!(shift<=28);}}
fn patch(original: &[u8], source: &[u8], export: &str, donor_export: &str, expected: &str) -> Result<(Vec<u8>,serde_json::Value),Error> {
 Validator::new_with_features(WasmFeatures::all()).validate_all(original)?;
 let initial=wat::parse_bytes(source)?.into_owned();
 let has_memory=Parser::new(0).parse_all(&initial).any(|p|matches!(p,Ok(Payload::MemorySection(_))));
 // Historical WAT bodies intentionally omit their destination module's memory.
 // Add a validation-only memory declaration; function bytes remain unchanged.
 let generated=if has_memory {initial}else{
  let text=std::str::from_utf8(source)?;
  wat::parse_str(&text.replacen("(module", "(module (memory 1)", 1))?
 };
 Validator::new_with_features(WasmFeatures::all()).validate_all(&generated)?;
 let mut donor_types=vec![]; let mut donor_functions=vec![]; let mut donor_exports=vec![];
 for payload in Parser::new(0).parse_all(&generated) { match payload? {
  Payload::TypeSection(s) => for t in s.into_iter_err_on_gc_types() { donor_types.push(t?); },
  Payload::FunctionSection(s) => for t in s { donor_functions.push(t?); },
  Payload::ExportSection(s) => for e in s { let e=e?; if e.kind==wasmparser::ExternalKind::Func { donor_exports.push((e.name.to_owned(),e.index)); } },
  Payload::ImportSection(_) | Payload::GlobalSection(_) | Payload::TableSection(_) | Payload::DataSection(_) | Payload::ElementSection(_) | Payload::StartSection {..} => return Err("donor must be a standalone stateless kernel".into()),
  Payload::CodeSectionEntry(body) => {
   let mut ops=body.get_operators_reader()?;
   while !ops.eof() { match ops.read()? {
    wasmparser::Operator::Call {..} | wasmparser::Operator::CallIndirect {..} | wasmparser::Operator::GlobalGet {..} | wasmparser::Operator::GlobalSet {..} | wasmparser::Operator::RefFunc {..} => return Err("donor references module-local indices".into()), _=>{} }
   }
  }, _=>{}
 } }
 if donor_functions.len()!=1 || donor_exports!=[(donor_export.to_owned(),0)] { return Err("donor export/function count mismatch".into()); }
 let donor_type=&donor_types[donor_functions[0] as usize];
 if donor_type.params()!=[ValType::I32;9] || !donor_type.results().is_empty() { return Err("donor ABI must be 9xi32 -> ()".into()); }
 let mut imports=0;let mut index=None;let mut exported_functions=vec![];let mut types=vec![];let mut function_types=vec![];let mut bodies=vec![];
 for p in Parser::new(0).parse_all(&original){match p?{
  Payload::TypeSection(s)=>for t in s.into_iter_err_on_gc_types(){types.push(t?);},
  Payload::ImportSection(s)=>for i in s {if matches!(i?.ty,TypeRef::Func(_)){imports+=1;}},
  Payload::FunctionSection(s)=>for t in s{function_types.push(t?);},
  Payload::ExportSection(s)=>for e in s{let e=e?;if e.kind==wasmparser::ExternalKind::Func{exported_functions.push((e.index as usize,e.name.to_owned()));}if e.name==export{assert_eq!(e.kind,wasmparser::ExternalKind::Func);assert!(index.is_none());index=Some(e.index as usize);}},
  Payload::CodeSectionEntry(body)=>bodies.push(original[body.range()].to_vec()),_=>{}}}
 let index=index.ok_or("missing patch export")?;let aliases:Vec<_>=exported_functions.iter().filter(|e|e.0==index).map(|e|e.1.as_str()).collect();if aliases.len()!=1{return Err(format!("patch target has function aliases: {:?}",aliases).into());}assert!(index>=imports);let position=index-imports;let target=&types[function_types[position]as usize];if target.params()!=[ValType::I32;9] || !target.results().is_empty() {return Err("target ABI must be 9xi32 -> ()".into());}
 let mut replacement=None;for p in Parser::new(0).parse_all(&generated){if let Payload::CodeSectionEntry(body)=p?{assert!(replacement.is_none());replacement=Some(generated[body.range()].to_vec());}}
 let replacement=replacement.ok_or("missing generated body")?; if sha(&replacement)!=expected {return Err("kernel body hash mismatch".into());} let before=bodies[position].clone();if sha(&before)==expected {return Err("target already patched".into());} bodies[position]=replacement.clone();
 let mut code=leb(bodies.len());for b in bodies{code.extend(leb(b.len()));code.extend(b);}
 let mut result=original[..8].to_vec();let mut p=8;let mut replaced=0;
 while p<original.len(){let begin=p;let id=original[p];p+=1;let len=take_leb(&original,&mut p);let end=p+len;assert!(end<=original.len());if id==10{result.push(id);result.extend(leb(code.len()));result.extend(&code);replaced+=1;}else{result.extend(&original[begin..end]);}p=end;}
 assert_eq!(replaced,1);Validator::new_with_features(WasmFeatures::all()).validate_all(&result)?;

 let report=serde_json::json!({"export":export,"original_sha256":sha(&original),"source_sha256":sha(&source),"replacement_body_sha256":sha(&replacement),"original_body_sha256":sha(&before),"output_sha256":sha(&result),"output_bytes":result.len(),"imported_functions":imports,"function_index":index,"code_position":position,"params":"9xi32","results":[],"wasmparser_validation":true,"scope":"One named typed kernel body replaced; every other section/body preserved"});Ok((result,report))
}

#[derive(Deserialize)]
struct Kernel { file: String, export: String, donor_export: String, sha256: String, body_sha256: String }
#[derive(Deserialize)]
struct Manifest { version: u32, kernels: Vec<Kernel> }
fn build(original:&[u8], directory:&Path) -> Result<(Vec<u8>,Vec<serde_json::Value>),Error> {
 let manifest:Manifest=serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
 if manifest.version!=1 || manifest.kernels.len()!=34 {return Err("expected manifest version 1 with exactly 34 kernels".into());}
 let mut expected=std::collections::BTreeSet::new();
 for k in &manifest.kernels { if !k.export.starts_with("__imajev_") || !expected.insert(k.export.clone()) {return Err("duplicate or invalid kernel export".into());} }
 let mut actual=std::collections::BTreeSet::new();
 for p in Parser::new(0).parse_all(original) {if let Payload::ExportSection(s)=p? {for e in s {let e=e?;if e.name.starts_with("__imajev_") {actual.insert(e.name.to_owned());}}}}
 if actual!=expected {return Err(format!("kernel coverage mismatch; missing {:?}; extra {:?}",expected.difference(&actual).collect::<Vec<_>>(),actual.difference(&expected).collect::<Vec<_>>()).into());}
 let mut module=original.to_vec();let mut reports=vec![];
 for k in manifest.kernels {
  if Path::new(&k.file).components().count()!=1 || !k.file.ends_with(".wat.gz") {return Err("kernel path must be one compressed WAT filename".into());}
  let source=decode_source(&fs::read(directory.join(&k.file))?)?;
  if sha(&source)!=k.sha256 {return Err(format!("kernel source checksum mismatch: {}",k.file).into());}
  let (next,report)=patch(&module,&source,&k.export,&k.donor_export,&k.body_sha256)?;module=next;reports.push(report);
 }
 Ok((module,reports))
}
fn decode_source(encoded:&[u8])->Result<Vec<u8>,Error> {
 use std::io::Read;
 const MAX:usize=4*1024*1024;
 let mut source=Vec::new();
 flate2::read::GzDecoder::new(encoded).take((MAX+1) as u64).read_to_end(&mut source)?;
 if source.len()>MAX {return Err("WAT source exceeds size bound".into());}
 Ok(source)
}
fn main()->Result<(),Error> {
 let a:Vec<_>=env::args().collect();
 if a.len()!=4 {return Err("usage: imajev-wasm-kernel-patch RAW.wasm KERNEL_DIRECTORY OUTPUT.wasm".into());}
 let output=Path::new(&a[3]);if output.exists(){return Err("refusing to overwrite output".into());}
 let (module,reports)=build(&fs::read(&a[1])?,Path::new(&a[2]))?;
 use std::io::Write;let mut file=fs::OpenOptions::new().write(true).create_new(true).open(output)?;file.write_all(&module)?;
 println!("{}",serde_json::json!({"module":sha(&module),"patches":reports,"wasmparser_validation":true,"complete_kernel_coverage":true}));Ok(())
}

#[cfg(test)]
mod tests {
 use super::*;
 fn body_hash(text:&str)->String {let b=wat::parse_str(text).unwrap();for p in Parser::new(0).parse_all(&b) {if let Payload::CodeSectionEntry(body)=p.unwrap(){return sha(&b[body.range()]);}}panic!("body");}
 fn target()->Vec<u8> {wat::parse_str("(module (memory 1) (func (export \"kernel\") (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (local.get 0) (i32.const 0) (i32.store)))").unwrap()}
 const SOURCE:&str="(module (func (export \"donor\") (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (local.get 0) (i32.const 42) (i32.store)))";
 #[test] fn compressed_source_preserves_bytes_and_rejects_corruption_and_oversized_input() {
  use std::io::Write;
  let compress=|bytes:&[u8]| {let mut e=flate2::write::GzEncoder::new(Vec::new(),flate2::Compression::default());e.write_all(bytes).unwrap();e.finish().unwrap()};
  let encoded=compress(SOURCE.as_bytes());assert_eq!(decode_source(&encoded).unwrap(),SOURCE.as_bytes());
  assert!(decode_source(b"invalid gzip").is_err());
  assert!(decode_source(&encoded[..encoded.len()-4]).is_err());
  assert!(decode_source(&compress(&vec![b' ';4*1024*1024+1])).is_err());
 }
 #[test] fn validates_injected_memory_and_changes_only_expected_body() {
  let input=target();let (output,report)=patch(&input,SOURCE.as_bytes(),"kernel","donor",&body_hash(SOURCE)).unwrap();
  assert_ne!(input,output);assert_eq!(report["replacement_body_sha256"],body_hash(SOURCE));
  assert!(patch(&output,SOURCE.as_bytes(),"kernel","donor",&body_hash(SOURCE)).is_err());
 }
 #[test] fn refuses_wrong_names_types_hashes_and_malformed_modules() {
  let input=target();let hash=body_hash(SOURCE);
  assert!(patch(&input,SOURCE.as_bytes(),"missing","donor",&hash).is_err());
  assert!(patch(&input,SOURCE.as_bytes(),"kernel","wrong",&hash).is_err());
  assert!(patch(&input,SOURCE.as_bytes(),"kernel","donor","wrong").is_err());
  assert!(patch(b"broken",SOURCE.as_bytes(),"kernel","donor",&hash).is_err());
  let bad=SOURCE.replace("i32 i32 i32 i32 i32 i32 i32 i32 i32", "i32");
  assert!(patch(&input,bad.as_bytes(),"kernel","donor",&body_hash(&bad)).is_err());
 }
 #[test] fn refuses_aliases_multiple_functions_and_module_local_state() {
  let alias=wat::parse_str("(module (memory 1) (func $f (export \"kernel\") (param i32 i32 i32 i32 i32 i32 i32 i32 i32)) (export \"alias\" (func $f)))").unwrap();
  assert!(patch(&alias,SOURCE.as_bytes(),"kernel","donor",&body_hash(SOURCE)).is_err());
  for extra in ["(func)","(global i32 (i32.const 1))"] {
   let text=SOURCE.replacen("(module", &format!("(module {extra}"),1);
   assert!(patch(&target(),text.as_bytes(),"kernel","donor",&body_hash(SOURCE)).is_err());
  }
 }
}
