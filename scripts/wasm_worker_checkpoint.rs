// Append an after-CDK worker checkpoint without modifying its original body.
use std::{env,fs};
use wasmparser::{Parser,Payload,TypeRef,ValType,ExternalKind,Validator,WasmFeatures};
use sha2::{Digest,Sha256};
fn leb(mut n:usize)->Vec<u8>{let mut b=vec![];loop{let v=(n&127)as u8;n>>=7;b.push(v|if n>0{128}else{0});if n==0{return b}}}
fn take(b:&[u8],p:&mut usize)->usize{let mut n=0;let mut s=0;loop{let v=b[*p];*p+=1;n|=((v&127)as usize)<<s;if v<128{return n}s+=7;assert!(s<=35)}}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<_>=env::args().collect();assert_eq!(a.len(),3);let original=fs::read(&a[1])?;
 let(mut types,mut funcs,mut bodies,mut exports)=(vec![],vec![],vec![],vec![]);let(mut imports,mut globals,mut pc)=(0usize,0usize,None);
 for p in Parser::new(0).parse_all(&original){match p?{
  Payload::TypeSection(s)=>for t in s.into_iter_err_on_gc_types(){types.push(t?)},
  Payload::ImportSection(s)=>for i in s{let i=i?;match i.ty{TypeRef::Func(t)=>{if i.module=="ic0"&&i.name=="performance_counter"{assert!(pc.is_none());let ty=&types[t as usize];assert_eq!(ty.params(),&[ValType::I32]);assert_eq!(ty.results(),&[ValType::I64]);pc=Some(imports)}imports+=1},TypeRef::Global(_)=>return Err("Imported globals are unsupported".into()),_=>{}}},
  Payload::FunctionSection(s)=>for t in s{funcs.push(t?)},
  Payload::GlobalSection(s)=>globals=s.count()as usize,
  Payload::ExportSection(s)=>for e in s{let e=e?;exports.push((e.name.to_owned(),e.kind,e.index as usize))},
  Payload::CodeSectionEntry(b)=>bodies.push(original[b.range()].to_vec()),_=>{}}}
 let find=|name:&str|->Result<usize,Box<dyn std::error::Error>>{let found:Vec<_>=exports.iter().filter(|e|e.0==name).collect();assert_eq!(found.len(),1);assert_eq!(found[0].1,ExternalKind::Func);Ok(found[0].2)};
 let worker=find("canister_update inference_step")?;let getter=find("__imajev_worker_checkpoint")?;let arm=find("__imajev_arm_worker_checkpoint")?;
 for index in [worker,getter,arm]{assert!(index>=imports);assert_eq!(exports.iter().filter(|e|e.1==ExternalKind::Func&&e.2==index).count(),1)}
 let wt=&types[funcs[worker-imports]as usize];assert!(wt.params().is_empty()&&wt.results().is_empty());
 let at=&types[funcs[arm-imports]as usize];assert!(at.params().is_empty()&&at.results().is_empty());
 let gt=&types[funcs[getter-imports]as usize];assert!(gt.params().is_empty());assert_eq!(gt.results(),&[ValType::I64]);
 let counter=globals;let armed=globals+1;let wrapper=imports+funcs.len();let old=bodies.clone();
 let mut get=vec![0,0x23];get.extend(leb(counter));get.push(0x0b);bodies[getter-imports]=get;
 let mut set=vec![0,0x41,1,0x24];set.extend(leb(armed));set.push(0x0b);bodies[arm-imports]=set;
 let mut wrap=vec![0,0x10];wrap.extend(leb(worker));wrap.push(0x23);wrap.extend(leb(armed));wrap.extend([0x04,0x40,0x41,0,0x10]);wrap.extend(leb(pc.ok_or("missing performance_counter")?));wrap.push(0x24);wrap.extend(leb(counter));wrap.extend([0x41,0,0x24]);wrap.extend(leb(armed));wrap.extend([0x0b,0x0b]);bodies.push(wrap.clone());
 let mut code=leb(bodies.len());for b in &bodies{code.extend(leb(b.len()));code.extend(b)}
 let mut es=leb(exports.len());for(name,kind,index)in &exports{es.extend(leb(name.len()));es.extend(name.as_bytes());es.push(match kind{ExternalKind::Func=>0,ExternalKind::Table=>1,ExternalKind::Memory=>2,ExternalKind::Global=>3,ExternalKind::Tag=>4});es.extend(leb(if name=="canister_update inference_step"{wrapper}else{*index}))}
 let mut out=original[..8].to_vec();let mut p=8;let mut changed=vec![];
 while p<original.len(){let begin=p;let id=original[p];p+=1;let len=take(&original,&mut p);let end=p+len;let b=&original[p..end];let mut payload=None;
  if id==3{let mut at=0;let n=take(b,&mut at);assert_eq!(n,funcs.len());let mut v=leb(n+1);v.extend(&b[at..]);v.extend(leb(funcs[worker-imports]as usize));payload=Some(v)}
  if id==6{let mut at=0;let n=take(b,&mut at);assert_eq!(n,globals);let mut v=leb(n+2);v.extend(&b[at..]);v.extend([0x7e,1,0x42,0,0x0b,0x7f,1,0x41,0,0x0b]);payload=Some(v)}
  if id==7{payload=Some(es.clone())}if id==10{payload=Some(code.clone())}
  if let Some(v)=payload{out.push(id);out.extend(leb(v.len()));out.extend(v);changed.push(id)}else{out.extend(&original[begin..end])}p=end;
 }
 assert_eq!(changed,vec![3,6,7,10]);Validator::new_with_features(WasmFeatures::all()).validate_all(&out)?;
 for(i,b)in old.iter().enumerate(){if i!=getter-imports&&i!=arm-imports{assert_eq!(b,&bodies[i])}}
 fs::write(&a[2],&out)?;let sha=|b:&[u8]|Sha256::digest(b).iter().map(|v|format!("{v:02x}")).collect::<String>();
 println!("{}",serde_json::json!({"complete":true,"input":sha(&original),"module":sha(&out),"worker_function":worker,"wrapper_function":wrapper,"wrapper_body":wrap,"wrapper_body_sha256":sha(&wrap),"getter_function":getter,"arm_function":arm,"original_worker_body_preserved":true,"all_other_original_bodies_preserved":true,"wasmparser_validation":true,"scope":"Counter checkpoint after original synchronous CDK handler returns. Only authenticated valid workers arm recording. Three store/constant opcodes and structural ends remain after the checkpoint; no proven whole-message upper bound claimed."}));Ok(())
}
