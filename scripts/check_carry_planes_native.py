#!/usr/bin/env python3
"""Verify Python Huffman carry framing against recorded canister/zlib bytes."""
import argparse,hashlib,json,os,pathlib,struct,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from mlp_delta_carry import encode_request,HUFFMAN_NAME
from transport import decode
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',default='artifacts/prefix_codec');ap.add_argument('--raw-threshold',type=float,default=0.);args=ap.parse_args()
reference=ROOT/'artifacts/prefix_codec';base=ROOT/args.directory;base.mkdir(parents=True,exist_ok=True)
if base!=reference:
    link=base/'mlp-delta-fusion-check-v2'
    if not link.exists():link.symlink_to(reference/'mlp-delta-fusion-check-v2',target_is_directory=True)
cases=[]
for layer in [0,1,3]:
    source=base/'mlp-delta-fusion-check-v2'/f'{layer:02d}-prepare-partial.response.bin'
    old=base/'mlp-delta-fusion-check-v2'/f'{layer:02d}-fused.request.bin'
    prefix=reference/'full-state-layout-proof/prefix/queries/states'/f'layer-{layer+1:02d}.npz'
    raw=old.read_bytes();header=json.loads(raw[4:4+struct.unpack('<I',raw[:4])[0]])
    _,state=decode(source.read_bytes())
    with np.load(prefix,allow_pickle=False)as p:conv=p['conv'].copy();log=p['delta_log'].copy()
    header['encoding']=HUFFMAN_NAME;frame=encode_request(header,state,conv,log,raw_threshold=args.raw_threshold)
    dest=base/f'carry-huffman-{layer:02d}.request.bin';dest.write_bytes(frame)
    cases.append(dict(layer=layer,frame_bytes=len(frame),frame_sha256=hashlib.sha256(frame).hexdigest(),source_reply_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),old_frame_sha256=hashlib.sha256(raw).hexdigest(),prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest()))
log=base/'native-test.log'
with log.open('w')as out:
    result=subprocess.run(['cargo','test','--offline','-p','imajev-runtime','--features','experimental-mlp-delta-fusion,experimental-blake3','python_huffman_matches_recorded_deflate_bytes','--','--ignored'],cwd=ROOT,env=dict(os.environ,CARRY_FIXTURE_ROOT=str(base)),stdout=out,stderr=out)
report=dict(exit_code=result.returncode,raw_threshold=args.raw_threshold,cases=cases,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/"client/mlp_delta_carry.py",ROOT/"crates/imajev-runtime/src/carry_planes.rs",ROOT/"crates/imajev-runtime/src/mlp_delta_fusion.rs"]},script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(base/'carry-huffman-native-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(result.returncode)
