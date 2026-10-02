#!/usr/bin/env python3
"""Re-encode historical journals to isolate codec size, without claiming query timing."""
import argparse,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import decode,encode
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/full-int8-canister');ap.add_argument('--output',default='docs/wire-codec-analysis.json');args=ap.parse_args();directory=ROOT/args.directory
report=json.loads((directory/'first-report.json').read_text());old=0;new=0;records=[];start=time.perf_counter()
for q in report['queries']:
 sizes={}
 for side,key in [('request','request_bytes'),('response','reply_bytes')]:
  raw=(directory/'queries'/f"{q['index']:06d}.{side}.bin").read_bytes();h,v=decode(raw);h['encoding']='bf16-exact';packed=encode(h,v);_,restored=decode(packed)
  assert np.array_equal(v.view(np.uint32),restored.view(np.uint32))
  overhead=q['ok'][key]-len(raw);sizes[side]=dict(original_binary_bytes=len(raw),compressed_binary_bytes=len(packed),original_candid_overhead=overhead)
  old+=q['ok'][key];new+=len(packed)+overhead
 records.append(dict(index=q['index'],op=q['op'],**sizes))
result=dict(scope='Offline journal re-encoding, same shapes and values; Candid overhead held fixed, not actual canister timing/instructions',old_candid_bytes=old,estimated_codec_only_candid_bytes=new,reduction_fraction=1-new/old,queries=len(records),every_value_bitwise_equal=True,wall_seconds=time.perf_counter()-start,records=records)
(ROOT/args.output).write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'})
