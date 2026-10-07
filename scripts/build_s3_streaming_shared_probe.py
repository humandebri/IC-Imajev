#!/usr/bin/env python3
"""Build full projection diagnostic with verified shared-helper kernel."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 original=ROOT/'artifacts/s3-k2-batch48-shared-probe-v1/frozen-builder.py'
 d=ROOT/'artifacts/s3-streaming-shared-probe-v1';d.mkdir(exist_ok=False)
 source=original.read_text().replace('s3-k2-batch48-shared-probe-v1','s3-streaming-shared-probe-v1').replace('s3-k2-batch48-kernels-v1','s3-streaming-shared-kernels-v1')
 before="kr['kernels']=[k for k in kr['kernels'] if not k['seed']];audit=json.loads((kernels/'layout-audit.json').read_text());validation=json.loads((kernels/'wasm-validation.json').read_text());assert audit['complete']and validation['all2_wasm_validated']"
 after="kr['kernels']=[dict(path='artifacts/s3-streaming-shared-kernels-v1/256.wat',symbol='__imajev_s3_streaming_256',locals=kr['main_locals'])];audit=json.loads((kernels/'execution-v2-report.json').read_text());validation=json.loads((kernels/'patch-validation.json').read_text());assert audit['complete']and validation['complete']"
 assert source.count(before)==1;source=source.replace(before,after)
 source=source.replace("kernels/'plan.py'","ROOT/'artifacts/rank343-mixed-scheme-screen-v1/WWC.py'")
 source=source.replace('rows%128!=0','rows%256!=0').replace('step_by(128)','step_by(256)').replace('wp:[*const i16;4]','wp:[*const i16;8]').replace('tile128','tile256').replace('n*128{{','n*256{{').replace('output_tile=128','output_tile=256')
 before="str(ROOT/'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique')"
 after="str(ROOT/('artifacts/wasm-audit-target/release/imajev-wasm-patch-shared'if symbol=='__imajev_s3_streaming_256'else'artifacts/wasm-audit-target/release/imajev-wasm-patch-unique'))"
 assert source.count(before)==1;source=source.replace(before,after)
 source=source.replace("kernels/'layout-audit.json'","kernels/'execution-v2-report.json'").replace("kernels/'wasm-validation.json'","kernels/'patch-validation.json'")
 frozen=d/'frozen-builder.py';frozen.write_text(source)
 (d/'builder-entry-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),original,frozen]},indent=2)+'\n')
 exec(compile(source,str(frozen),'exec'),dict(__file__=__file__,__name__='__main__'))
if __name__=='__main__':main()
