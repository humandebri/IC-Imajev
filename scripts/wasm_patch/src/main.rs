use std::{env,fs};
use wasmparser::{Parser,Payload,TypeRef,ValType,Validator,WasmFeatures};
use sha2::{Digest,Sha256};
fn leb(mut v:usize)->Vec<u8>{let mut b=vec![];loop{let x=(v&127)as u8;v>>=7;b.push(x|if v!=0{128}else{0});if v==0{return b;}}}
fn take_leb(b:&[u8],p:&mut usize)->usize{let mut x=0usize;let mut shift=0;loop{let v=b[*p];*p+=1;x|=((v&127)as usize)<<shift;if v&128==0{return x;}shift+=7;assert!(shift<=28);}}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<_>=env::args().collect();assert!(a.len()==4 || a.len()==5);let export=a.get(4).map(String::as_str).unwrap_or("__imajev_pair_accumulate");let original=fs::read(&a[1])?;let source=fs::read(&a[2])?;let generated=wat::parse_bytes(&source)?.into_owned();
 let mut imports=0;let mut index=None;let mut types=vec![];let mut function_types=vec![];let mut bodies=vec![];
 for p in Parser::new(0).parse_all(&original){match p?{
  Payload::TypeSection(s)=>for t in s.into_iter_err_on_gc_types(){types.push(t?);},
  Payload::ImportSection(s)=>for i in s {if matches!(i?.ty,TypeRef::Func(_)){imports+=1;}},
  Payload::FunctionSection(s)=>for t in s{function_types.push(t?);},
  Payload::ExportSection(s)=>for e in s{let e=e?;if e.name==export{assert_eq!(e.kind,wasmparser::ExternalKind::Func);assert!(index.is_none());index=Some(e.index as usize);}},
  Payload::CodeSectionEntry(body)=>bodies.push(original[body.range()].to_vec()),_=>{}}}
 let index=index.ok_or("missing patch export")?;assert!(index>=imports);let position=index-imports;let target=&types[function_types[position]as usize];assert_eq!(target.params(),&[ValType::I32;9]);assert!(target.results().is_empty());
 let mut replacement=None;for p in Parser::new(0).parse_all(&generated){if let Payload::CodeSectionEntry(body)=p?{assert!(replacement.is_none());replacement=Some(generated[body.range()].to_vec());}}
 let replacement=replacement.ok_or("missing generated body")?;let before=bodies[position].clone();bodies[position]=replacement.clone();
 let mut code=leb(bodies.len());for b in bodies{code.extend(leb(b.len()));code.extend(b);}
 let mut result=original[..8].to_vec();let mut p=8;let mut replaced=0;
 while p<original.len(){let begin=p;let id=original[p];p+=1;let len=take_leb(&original,&mut p);let end=p+len;assert!(end<=original.len());if id==10{result.push(id);result.extend(leb(code.len()));result.extend(&code);replaced+=1;}else{result.extend(&original[begin..end]);}p=end;}
 assert_eq!(replaced,1);Validator::new_with_features(WasmFeatures::all()).validate_all(&result)?;
 fs::write(&a[3],&result)?;let sha=|b:&[u8]|Sha256::digest(b).iter().map(|v|format!("{v:02x}")).collect::<String>();
 println!("{}",serde_json::json!({"export":export,"original_sha256":sha(&original),"source_sha256":sha(&source),"replacement_body_sha256":sha(&replacement),"original_body_sha256":sha(&before),"output_sha256":sha(&result),"output_bytes":result.len(),"imported_functions":imports,"function_index":index,"code_position":position,"params":"9xi32","results":[],"wasmparser_validation":true,"scope":"One named diagnostic function body replaced; every other section/body preserved"}));Ok(())
}
