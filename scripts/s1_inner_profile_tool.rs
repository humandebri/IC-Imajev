use std::{env, fs};
#[cfg(imports_only)]use wasmparser::{Parser,Payload,TypeRef};
#[cfg(not(imports_only))]
use candid::{CandidType,Decode};
#[cfg(not(imports_only))]
use serde::Deserialize;

#[cfg(not(imports_only))]#[derive(CandidType,Deserialize)]
struct ProfileReply {inner_profile: Vec<u64>}

fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=env::args().collect();
    #[cfg(not(imports_only))] {
        assert_eq!(args[1],"decode");
        let raw=fs::read_to_string(&args[2])?;
        let text=raw.trim();
        let bytes=(0..text.len()).step_by(2).map(|i|u8::from_str_radix(&text[i..i+2],16)).collect::<Result<Vec<_>,_>>()?;
        let reply=Decode!(&bytes,ProfileReply)?;
        println!("{}",serde_json::to_string(&reply.inner_profile)?);
        return Ok(());
    }
    #[cfg(imports_only)] {
    assert_eq!(args[1],"imports");
    let bytes=fs::read(&args[2])?;
    let mut types=Vec::new();let mut imports=Vec::new();
    for p in Parser::new(0).parse_all(&bytes) {
        match p? {
            Payload::TypeSection(section)=>for t in section.into_iter_err_on_gc_types(){types.push(t?);},
            Payload::ImportSection(section)=>for i in section {
                let i=i?;
                if let TypeRef::Func(t)=i.ty {
                    let t=&types[t as usize];
                    imports.push(serde_json::json!({"module":i.module,"name":i.name,"params":t.params().iter().map(|v|v.to_string()).collect::<Vec<_>>(),"results":t.results().iter().map(|v|v.to_string()).collect::<Vec<_>>()}));
                }
            },
            _=>{}
        }
    }
    println!("{}",serde_json::to_string(&imports)?);
    Ok(())
    }
}
