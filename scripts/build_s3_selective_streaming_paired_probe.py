#!/usr/bin/env python3
"""Compare current control and both streaming bodies on identical fixed state."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-streaming-shared-probe-v1/frozen-builder.py'
 d=ROOT/'artifacts/s3-selective-streaming-paired-probe-v1';d.mkdir(exist_ok=False)
 selected=ROOT/'artifacts/s3-selective-streaming-kernels-v1'
 proof=json.loads((selected/'patch-validation.json').read_text());assert proof['complete']
 for p,h in proof['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 text=(selected/'256.wat').read_text();assert text.count('(export "__imajev_s3_streaming_256")')==1
 derived=d/'256-selective.wat';derived.write_text(text.replace('(export "__imajev_s3_streaming_256")','(export "__imajev_s3_streaming_256_selective")'))
 source=original.read_text().replace('s3-streaming-shared-probe-v1','s3-selective-streaming-paired-probe-v1')
 before="kr['kernels']=[dict(path='artifacts/s3-streaming-shared-kernels-v1/256.wat',symbol='__imajev_s3_streaming_256',locals=kr['main_locals'])]"
 after=before+";kr['kernels'].append(dict(path='artifacts/s3-selective-streaming-paired-probe-v1/256-selective.wat',symbol='__imajev_s3_streaming_256_selective',locals=8956))"
 assert source.count(before)==1;source=source.replace(before,after)
 before='pub fn project_wide(&self,q:&QuantizedRows';assert source.count(before)==1;source=source.replace(before,'pub fn project_wide(&self,selective:bool,q:&QuantizedRows')
 before='let f=crate::s2_kernel::tile256;';assert source.count(before)==1;source=source.replace(before,'let f=if selective{crate::s2_kernel::tile256_selective}else{crate::s2_kernel::tile256};')
 before=" (src/'lib.rs').write_text(lib)";assert source.count(before)==1
 source=source.replace(before," lib=lib.replace('assert!(method==3||method==4)','assert!(method==3||method==4||method==5)').replace('(method==4).then','(method==4||method==5).then').replace('if method==4{f.win','if method==4||method==5{f.win').replace('.project_wide(&q,','.project_wide(method==5,&q,')\n"+before)
 before="suffix='_seed'if '_seed'in k['symbol']else ''";assert source.count(before)==1;source=source.replace(before,"suffix='_selective'if k['symbol'].endswith('_selective')else ''")
 source=source.replace("symbol=='__imajev_s3_streaming_256'","symbol.startswith('__imajev_s3_streaming_256')")
 before="files=[Path(__file__),kernels/";assert source.count(before)==1;source=source.replace(before,"files=[Path(__file__),ROOT/'artifacts/s3-selective-streaming-kernels-v1/patch-validation.json',ROOT/'artifacts/s3-selective-streaming-kernels-v1/execution-v2-report.json',kernels/")
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 files=[Path(__file__),original,frozen,derived,selected/'256.wat',selected/'patch-validation.json']
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
