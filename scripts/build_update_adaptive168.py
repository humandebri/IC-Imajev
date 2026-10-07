#!/usr/bin/env python3
"""Use exact adaptive168 only for measured4096/8192-row projections."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'artifacts/update-row-reuse-v2';d=ROOT/'artifacts/update-adaptive168-v1';d.mkdir(exist_ok=False)
 proof=ROOT/'artifacts/s1-adaptive168-v1/summary.json';r=json.loads(proof.read_text());assert r['native_bits_equal'] and r['native_digests_recomputed'] and r['locals']<10000
 for hashes in [json.loads((old/'workflow-hashes.json').read_text()),r['workflow_hashes']]:assert all(sha(ROOT/p)==h for p,h in hashes.items())
 for p in list(old.glob('*.rs'))+list(old.glob('*.wat')):(d/p.name).write_bytes(p.read_bytes())
 (d/'int8_168.wat').write_bytes((ROOT/'artifacts/s1-adaptive168-v1/build/kernel168.wat').read_bytes())
 s=(old/'frozen-builder.py').read_text().replace('artifacts/update-row-reuse-v2','artifacts/update-adaptive168-v1')
 patch=r"""    p=D/'runtime/strassen_raw.rs';text=p.read_text()
    before='let tile=if wide && rows-r>=160{160}else if wide && rows-r>=128{128}else{32};'
    assert text.count(before)==1
    after='let use168=wide && (rows==4096 || rows==8192);let tile=if use168 && rows-r>=168{168}else if !use168 && wide && rows-r>=160{160}else if wide && rows-r>=128{128}else{32};'
    text=text.replace(before,after)
    anchor='   if tile==160 {accumulate160('
    assert text.count(anchor)==1
    text=text.replace(anchor,'   if tile==168 {accumulate168(ap.as_ptr().cast(),wp.as_ptr().cast(),cols,block*256,q.scales().as_ptr().add(block),cols/256,s.as_ptr(),sums.as_mut_ptr(),q.rows());continue;}\n   #[cfg(feature="experimental-strassen-output128")]\n'+anchor)
    a=text.index('#[export_name="__imajev_s1_160_accumulate"]');b=text.index('#[cfg(test)]',a)
    stub=text[a:b].replace('__imajev_s1_160_accumulate','__imajev_s1_168_accumulate').replace('fn accumulate160(', 'fn accumulate168(').replace('0..n*160','0..n*168').replace('(marker^164)','(marker^168)')
    p.write_text(text[:b]+stub+text[b:])
"""
 anchor="    runtime = base['runtime_command'][:]";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 patch="""    wat=D.parent/'int8_168.wat';prior=D/'full.wasm';result=D/'candidate168.wasm'
    row=json.loads(subprocess.check_output([str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'),str(prior),str(wat),str(result),'__imajev_s1_168_accumulate'],text=True))
    assert row['wasmparser_validation'];patches.append(row);prior.rename(D/'before-int8-168.wasm');result.rename(D/'full.wasm')
"""
 anchor="    sources = [D/'finite-sites.json'";assert s.count(anchor)==1;s=s.replace(anchor,patch+anchor)
 (d/'frozen-builder.py').write_text(s)
 files=[Path(__file__),old/'workflow-hashes.json',old/'frozen-builder.py',proof,d/'frozen-builder.py']+list(d.glob('*.rs'))+list(d.glob('*.wat'))
 (d/'workflow-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in files},indent=2)+'\n')
 exec(compile(s,str(d/'frozen-builder.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
 changed=sorted(p.name for p in(old/'build/runtime').glob('*.rs')if p.read_bytes()!=(d/'build/runtime'/p.name).read_bytes());assert changed==['strassen_raw.rs'],changed
 (d/'runtime-comparison.json').write_text(json.dumps(dict(changed_runtime_files=changed,original_arithmetic_kernels_equal=True,added_kernel='__imajev_s1_168_accumulate',dispatch_rows=[4096,8192],scope='Same integer arithmetic and F32 order, wider register output tile only. All other shapes use original adaptive160.'),indent=2)+'\n')
if __name__=='__main__':main()
