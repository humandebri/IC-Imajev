#!/usr/bin/env python3
"""Replay selected actual query requests through the identical native pack reader."""
import argparse,json,pathlib,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'));from transport import decode
ap=argparse.ArgumentParser();ap.add_argument('--directory',default='artifacts/full-int8-canister');ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--pack',default='checkpoints/full-int8.pack');args=ap.parse_args();directory=ROOT/args.directory;queries=directory/'queries';seen=set();records=[]
for metric in sorted(queries.glob('*.metric.json')):
 data=json.loads(metric.read_text());op=data['op']
 if op in seen:continue
 seen.add(op);index=data['index'];request=queries/f'{index:06d}.request.bin';response=queries/f'{index:06d}.response.bin';native=directory/f'native-{op}.bin'
 subprocess.run([str(ROOT/'target/release/primitive'),str(request),str(native),str(ROOT/args.manifest),str(ROOT/args.pack)],check=True)
 _,a=decode(response.read_bytes());_,b=decode(native.read_bytes());assert a.shape==b.shape;error=abs(a-b);record=dict(op=op,index=index,bitwise_equal=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),max_error=float(error.max()),mean_error=float(error.mean()));records.append(record);print(record,flush=True)
 if op in ['embed','lora_project','linear_bf16','conv_state','add_bf16','swiglu_bf16','attention_gate']:assert record['bitwise_equal'],record
(directory/'native-wasm.json').write_text(json.dumps(dict(scope='Selected recorded actual query requests replayed in native Rust, same weights/inputs',cases=records),indent=2)+'\n')
