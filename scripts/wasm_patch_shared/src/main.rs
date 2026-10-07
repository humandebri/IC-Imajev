use std::{env,fs};
use wasmparser::{Parser,Payload,TypeRef,ValType,Validator,WasmFeatures,Operator};
use sha2::{Digest,Sha256};
fn leb(mut v:usize)->Vec<u8>{let mut b=vec![];loop{let x=(v&127)as u8;v>>=7;b.push(x|if v!=0{128}else{0});if v==0{return b;}}}
fn take(b:&[u8],p:&mut usize)->usize{let mut x=0;let mut shift=0;loop{let v=b[*p];*p+=1;x|=((v&127)as usize)<<shift;if v&128==0{return x;}shift+=7;assert!(shift<=28);}}
fn sha(b:&[u8])->String{Sha256::digest(b).iter().map(|v|format!("{v:02x}")).collect()}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<_>=env::args().collect();assert_eq!(a.len(),5);let original=fs::read(&a[1])?;let source=fs::read(&a[2])?;let generated=wat::parse_bytes(&source)?.into_owned();let export=&a[4];
 let mut imports=0usize;let mut target=None;let mut aliases=vec![];let mut types=vec![];let mut ft=vec![];let mut bodies=vec![];
 for p in Parser::new(0).parse_all(&original){match p?{
  Payload::TypeSection(s)=>for t in s.into_iter_err_on_gc_types(){types.push(t?);},
  Payload::ImportSection(s)=>for i in s{if matches!(i?.ty,TypeRef::Func(_)){imports+=1;}},
  Payload::FunctionSection(s)=>for t in s{ft.push(t?);},
  Payload::ExportSection(s)=>for e in s{let e=e?;if e.kind==wasmparser::ExternalKind::Func{aliases.push((e.index as usize,e.name.to_owned()));if e.name==export{assert!(target.is_none());target=Some(e.index as usize);}}},
  Payload::CodeSectionEntry(b)=>bodies.push(original[b.range()].to_vec()),_=>{}}}
 let target=target.ok_or("target missing")?;assert!(target>=imports);assert_eq!(aliases.iter().filter(|e|e.0==target).count(),1);let position=target-imports;
 assert_eq!(types[ft[position]as usize].params(),&[ValType::I32;9]);assert!(types[ft[position]as usize].results().is_empty());
 let mut gt=vec![];let mut gf=vec![];let mut gb=vec![];let mut replacement=None;let mut helper=None;let helper_index=imports+bodies.len();let mut relocated=0;
 for p in Parser::new(0).parse_all(&generated){match p?{
  Payload::TypeSection(s)=>for t in s.into_iter_err_on_gc_types(){gt.push(t?);},
  Payload::ImportSection(_)=>return Err("generated imports forbidden".into()),
  Payload::FunctionSection(s)=>for t in s{gf.push(t?);},
  Payload::ExportSection(s)=>for e in s{let e=e?;if e.name==export{assert_eq!(e.index,0);assert_eq!(e.kind,wasmparser::ExternalKind::Func);}},
  Payload::CodeSectionEntry(body)=>{
   let mut code=generated[body.range()].to_vec();let mut reader=body.get_operators_reader()?;let mut changes=vec![];
   while !reader.eof(){let begin=reader.original_position();let op=reader.read()?;let end=reader.original_position();match op{
    Operator::Call{function_index}=>{assert_eq!(gb.len(),0);assert_eq!(function_index,1);let mut bytes=vec![0x10];bytes.extend(leb(helper_index));changes.push((begin-body.range().start,end-body.range().start,bytes));relocated+=1;},
    Operator::RefFunc{..}|Operator::CallIndirect{..}|Operator::ReturnCall{..}|Operator::ReturnCallIndirect{..}=>return Err("unsupported reference/call relocation".into()),_=>{}}
   }
   for (begin,end,bytes)in changes.into_iter().rev(){code.splice(begin..end,bytes);}
   if gb.is_empty(){replacement=Some(code.clone());}else{assert_eq!(gb.len(),1);helper=Some(code.clone());}gb.push(code);
  },_=>{}}}
 assert_eq!(gb.len(),2);assert_eq!(gf.len(),2);assert_eq!(gt[gf[0]as usize].params(),&[ValType::I32;9]);assert!(gt[gf[0]as usize].results().is_empty());let ht=&gt[gf[1]as usize];assert_eq!(ht.params(),&[ValType::I32;9]);assert!(!ht.results().is_empty());assert!(ht.results().iter().all(|v|*v==ValType::V128));assert!(relocated>0);
 let before=bodies[position].clone();let replacement=replacement.unwrap();let helper=helper.unwrap();bodies[position]=replacement.clone();bodies.push(helper.clone());
 let mut code=leb(bodies.len());for b in bodies{code.extend(leb(b.len()));code.extend(b);}
 let mut helper_type=vec![0x60];helper_type.extend(leb(9));helper_type.extend([0x7f;9]);helper_type.extend(leb(ht.results().len()));helper_type.extend(vec![0x7b;ht.results().len()]);
 let mut result=original[..8].to_vec();let mut p=8;let mut replaced=vec![];
 while p<original.len(){let begin=p;let id=original[p];p+=1;let len=take(&original,&mut p);let end=p+len;assert!(end<=original.len());let payload=match id{
  1=>{let mut q=p;let count=take(&original,&mut q);assert_eq!(count,types.len());let mut data=leb(count+1);data.extend(&original[q..end]);data.extend(&helper_type);Some(data)},
  3=>{let mut q=p;let count=take(&original,&mut q);assert_eq!(count,ft.len());let mut data=leb(count+1);data.extend(&original[q..end]);data.extend(leb(types.len()));Some(data)},
  10=>Some(code.clone()),_=>None};
  if let Some(data)=payload{result.push(id);result.extend(leb(data.len()));result.extend(data);replaced.push(id);}else{result.extend(&original[begin..end]);}p=end;
 }
 assert_eq!(replaced,vec![1,3,10]);Validator::new_with_features(WasmFeatures::all()).validate_all(&result)?;fs::write(&a[3],&result)?;
 println!("{}",serde_json::json!({"export":export,"original_sha256":sha(&original),"source_sha256":sha(&source),"output_sha256":sha(&result),"output_bytes":result.len(),"original_body_sha256":sha(&before),"replacement_body_sha256":sha(&replacement),"helper_body_sha256":sha(&helper),"helper_function_index":helper_index,"helper_type_index":types.len(),"helper_results":ht.results().len(),"relocated_calls":relocated,"wasmparser_validation":true,"scope":"Replace one unique nine-I32 export, append one helper type/function/body, relocate only direct generated helper calls. Other original bytes preserved."}));Ok(())
}
