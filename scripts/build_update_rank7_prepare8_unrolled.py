#!/usr/bin/env python3
"""Latest validated prepare8 parent, with exactly the measured unrolled helper."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    original=ROOT/'scripts/build_update_rank7_prepare8.py'
    parent=ROOT/'artifacts/update-rank7-prepare8-v1/build'
    r=json.loads((parent/'report.json').read_text());assert sha(original)==r['source_hashes'][str(original.relative_to(ROOT))]
    probe=ROOT/'artifacts/rank7-prepare8-unrolled-probe-v1'
    audit=json.loads((probe/'independent-audit.json').read_text());assert audit['all_nonempty_measured_cases_improve']
    measured=(probe/'probe.rs').read_text();begin=measured.index('#[no_mangle]pub unsafe extern "C" fn prepare16(');end=measured.index('\n#[repr(align(16))]struct Aligned7',begin)
    helper=measured[begin:end].replace('#[no_mangle]pub unsafe extern "C" fn prepare16','#[cfg(target_arch="wasm32")]#[inline(always)]unsafe fn fill_unrolled')
    helper=helper.replace('{\n','{\n use core::arch::wasm32::*;\n',1)
    # Keep the scalar host reference byteexact. Replace only the wasm branch.
    code=original.read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path('+repr(str(ROOT))+')')
    code=code.replace('artifacts/update-common-raw-hybrid-v1/build','artifacts/update-rank7-prepare8-v1/build')
    code=code.replace('artifacts/rank7-prepare8-probe-v1','artifacts/rank7-prepare8-unrolled-probe-v1')
    code=code.replace('artifacts/update-rank7-prepare8-v1\';b=d', 'artifacts/update-rank7-prepare8-unrolled-v1\';b=d')
    start=code.index('    for a,z,count in [');finish=code.index('    p.write_text(s)',start)
    replacement='''    a=s.index('  #[cfg(target_arch="wasm32")]\\n  {use core::arch::wasm32::*;',s.index('unsafe fn prepare('))
    z=s.index('  #[cfg(not(target_arch="wasm32"))]',a)
    # The existing pair/block loops own this wasm branch. The helper's own
    # pair/block loops replace those outer loops below, avoiding duplicate work.
    prepare_begin=s.index('unsafe fn prepare(')
    prepare_end=s.index('\\npub(crate) fn index',prepare_begin)
    old_prepare=s[prepare_begin:prepare_end]
    native=old_prepare[old_prepare.index('  #[cfg(not(target_arch="wasm32"))]'):]
    new_prepare='unsafe fn prepare(q:&QuantizedRows,pairs:usize,out:*mut i16,first:usize,end:usize){\\n let cols=q.cols();\\n #[cfg(target_arch="wasm32")]fill_unrolled(q.values().as_ptr(),cols,pairs,out,first,end);\\n #[cfg(not(target_arch="wasm32"))]for pair in 0..pairs{for block in first..end{\\n'+native+'\\n'
    s=s[:prepare_begin]+new_prepare+HELPER+'\\n'+s[prepare_end:]
'''
    code=code[:start]+replacement+code[finish:]
    code='HELPER='+repr(helper)+'\n'+code
    d=ROOT/'artifacts/update-rank7-prepare8-unrolled-v1';d.mkdir(exist_ok=False)
    frozen=d/'frozen-builder.py';frozen.write_text(code)
    (d/'entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),original,frozen,probe/'probe.rs']},indent=2)+'\n')
    exec(compile(code,str(frozen),'exec'),dict(__file__=str(frozen),__name__='__main__'))
if __name__=='__main__':main()
