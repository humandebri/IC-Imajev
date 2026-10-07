#!/usr/bin/env python3
"""Bulk-append exact F32 bytes in carry encoders, preserving field order and drops."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-bf16-predicates-v1';d=ROOT/'artifacts/update-f32-byte-append-v1';proof=ROOT/'artifacts/f32-byte-append-v1/summary.json';r=json.loads(proof.read_text());assert r['complete'] and r['all_predicates_equal'] and r['saved_candid_redecoded'];assert all(sha(ROOT/p)==h for p,h in r['workflow_hashes'].items())
 assert all(sha(ROOT/p)==h for p,h in json.loads((old/'workflow-hashes.json').read_text()).items())
 d.mkdir(exist_ok=False)
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 helper=ROOT/'scripts/f32_byte_append.rs';(d/'f32_byte_append.rs').write_bytes(helper.read_bytes())
 replacements={
 'mlp_stream.rs':{
 'for v in &self.q.scales()[..n*C/256] {p.extend(v.to_le_bytes());}':'crate::append_f32_le(&mut p,&self.q.scales()[..n*C/256]);',
 'for v in self.ax {p.extend(v.to_le_bytes());}':'crate::append_f32_le(&mut p,&self.ax);drop(self.ax);',
 'for v in self.scales.into_iter().chain(self.down) {p.extend(v.to_le_bytes());}':'crate::append_f32_le(&mut p,&self.scales);drop(self.scales);crate::append_f32_le(&mut p,&self.down);drop(self.down);',
 'for v in &x[cursor..cursor + tail] {\n        p.extend_from_slice(&v.to_le_bytes());\n    }':'crate::append_f32_le(&mut p,&x[cursor..cursor + tail]);',
 'for v in &x[cursor..] {\n        p.extend_from_slice(&v.to_le_bytes());\n    }':'crate::append_f32_le(&mut p,&x[cursor..]);'},
 'projection_codec.rs':{'for v in &x[prefix+q..] {b.extend_from_slice(&v.to_le_bytes());}':'crate::append_f32_le(b,&x[prefix+q..]);'},
 'mlp_delta_stream.rs':{
 'for v in prep.q.scales()[..n*C/256].iter().chain(&prep.qa).chain(&prep.za).chain(&prep.gates).chain(base).chain(ax) {\n        payload.extend(v.to_le_bytes());\n    }':'crate::append_f32_le(&mut payload,&prep.q.scales()[..n*C/256]);crate::append_f32_le(&mut payload,&prep.qa);crate::append_f32_le(&mut payload,&prep.za);crate::append_f32_le(&mut payload,&prep.gates);crate::append_f32_le(&mut payload,base);crate::append_f32_le(&mut payload,ax);',
 'for v in &x[cursor..cursor + tail] {\n        b.extend_from_slice(&v.to_le_bytes());\n    }':'crate::append_f32_le(b,&x[cursor..cursor + tail]);'},
 'mlp_pipeline.rs':{'for v in &x[count+q..] {b.extend_from_slice(&v.to_le_bytes());}':'crate::append_f32_le(b,&x[count+q..]);'},
 'attention_mlp_stream.rs':{
 'for v in &v[qend..send] {\n            b.extend(v.to_le_bytes());\n        }':'crate::append_f32_le(b,&v[qend..send]);',
 'for v in &v[gend..] {\n            b.extend(v.to_le_bytes());\n        }':'crate::append_f32_le(b,&v[gend..]);'},
 'lib.rs':{
 'for v in x {\n                b.extend_from_slice(&v.to_le_bytes());\n            }':'crate::append_f32_le(&mut b,x);',
 'for v in &x[prefix..] {\n                b.extend_from_slice(&v.to_le_bytes());\n            }':'crate::append_f32_le(&mut b,&x[prefix..]);'},
 'prefix_hybrid_codec.rs':{'for v in &log[n*2048+t*4096+h*128..n*2048+t*4096+(h+1)*128] {packet.extend(v.to_le_bytes());}':'crate::append_f32_le(&mut packet,&log[n*2048+t*4096+h*128..n*2048+t*4096+(h+1)*128]);'},
 }
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-bf16-predicates-v1','artifacts/update-f32-byte-append-v1')
 patch="    replacements="+repr(replacements)+"\n"+'''    sites={}
    for name,items in replacements.items():
        p=D/'runtime'/name;text=p.read_text()
        for before,after in items.items():assert text.count(before)==1,(name,before);text=text.replace(before,after)
        sites[name]=len(items);p.write_text(text)
    assert sum(sites.values())==14
    (D/'f32-byte-append-sites.json').write_text(json.dumps(sites,indent=2)+'\\n')
    (D/'runtime/f32_byte_append.rs').write_bytes((D.parent/'f32_byte_append.rs').read_bytes())
    p=D/'runtime/lib.rs';p.write_text(p.read_text()+'\\nmod f32_byte_append;\\npub use f32_byte_append::append as append_f32_le;\\n')
'''
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor);(d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),proof,helper,old/'workflow-hashes.json',old/'frozen-builder.py',d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'));(d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==sorted(replacements),changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,added_runtime_files=['f32_byte_append.rs'],arithmetic_kernels_equal=True,scope='Fourteen scalar F32 byte-append loops replaced by twenty exact borrowed bulk copies. Field order, explicit owned Vec drops, validators and little endian representation preserved. Native helper keeps scalar byte conversion.'),indent=2)+'\n')
if __name__=='__main__':main()
