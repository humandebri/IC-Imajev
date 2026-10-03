use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
};
use wasmparser::{Name, Operator, Parser, Payload};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    if a[1] == "wat" {
        fs::write(&a[3], wat::parse_file(&a[2])?)?;
        return Ok(());
    }
    let include_all = a.get(2).is_some_and(|v| v == "--all");
    let bytes = fs::read(&a[1])?;
    let mut names = HashMap::new();
    let mut imports = 0;
    let mut index = 0;
    let mut bodies = vec![];
    for item in Parser::new(0).parse_all(&bytes) {
        match item? {
            Payload::ImportSection(r) => {
                for item in r {
                    if matches!(item?.ty, wasmparser::TypeRef::Func(_)) {
                        imports += 1;
                    }
                }
            }
            Payload::CustomSection(c) if c.name() == "name" => {
                for section in wasmparser::NameSectionReader::new(wasmparser::BinaryReader::new(
                    c.data(),
                    c.data_offset(),
                )) {
                    if let Name::Function(r) = section? {
                        for item in r {
                            let n = item?;
                            names.insert(n.index, n.name.to_string());
                        }
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let locals: u32 = body
                    .get_locals_reader()?
                    .into_iter()
                    .map(|x| x.unwrap().0)
                    .sum();
                let mut ops: BTreeMap<String, u64> = BTreeMap::new();
                let mut dots = 0;
                let mut calls: BTreeMap<u32,u64> = BTreeMap::new();
                let mut r = body.get_operators_reader()?;
                while !r.eof() {
                    let op = r.read()?;
                    if matches!(op, Operator::I32x4DotI16x8S) {
                        dots += 1;
                    }
                    if let Operator::Call { function_index } = &op { *calls.entry(*function_index).or_default() += 1; }
                    let name = format!("{op:?}");
                    let key = name.split([' ', '{']).next().unwrap().to_string();
                    *ops.entry(key).or_default() += 1;
                }
                bodies.push((index + imports, locals, dots, ops, calls));
                index += 1;
            }
            _ => {}
        }
    }
    let rows:Vec<_>=bodies.into_iter().filter(|(_,_,d,_,_)|include_all || *d>0).map(|(idx,locals,dots,ops,calls)| {
        let mut row=serde_json::json!({"function":idx,"name":names.get(&idx),"locals":locals,"dots":dots,"operations":ops});
        if include_all {row["calls"]=serde_json::json!(calls.into_iter().map(|(id,count)|serde_json::json!({"function":id,"name":names.get(&id),"count":count})).collect::<Vec<_>>());}
        row
    }).collect();
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
