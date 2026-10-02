#!/usr/bin/env python3
"""Pinned official MLX one-question uncached reference; rotations=1, text only."""
import argparse,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
SERVER='a0134749e0900189c129cd6bb5000969f3b64bb5'
sys.path.insert(0,str(ROOT/f'vendor/imajev-{SERVER}/src'))
from vision_decision.backend import MLXDirect
from vision_decision.contracts import ChoiceField,Option
from vision_decision.scoring import compile_question
from vision_decision.calibration import TemperatureCalibrator

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serving',action='store_true');ap.add_argument('--first-only',action='store_true');ap.add_argument('--orders',action='store_true');ap.add_argument('--output',default='artifacts/reference.json');args=ap.parse_args()
    output=ROOT/args.output
    if output.exists():raise SystemExit('Refusing to overwrite existing output')
    bundle=ROOT/'artifacts/base-bundle.json';bundle.write_text(json.dumps({'path':str(ROOT/'checkpoints/base')}))
    engine=MLXDirect(str(bundle),adapter=str(ROOT/'checkpoints/adapter/mlx'))
    cal=TemperatureCalibrator.load(ROOT/'checkpoints/adapter/calibration.json')
    cases=json.loads((ROOT/'benchmarks/cases.json').read_text())
    if args.first_only:cases=cases[:1]
    records=[]
    original=engine._candidate_logits
    capture={}
    def hooked(hidden,ids,readout_indices=None):
        engine.mx.eval(hidden)
        capture['hidden']=hidden.astype(engine.mx.float32).tolist()
        return original(hidden,ids,readout_indices)
    engine._candidate_logits=hooked
    for c in cases:
        options=c['options']
        for offset in range(len(options) if args.orders else 1):
            order=options[offset:]+options[:offset]
            field=ChoiceField(id=c['id'].replace('-','_'),type='choice',question=c['question'],options=[Option(value=x) for x in order])
            header,choices,texts=compile_question(field,c['state'],engine.prompt_layout)
            labels=engine._labels(header,len(choices),0)
            prompt=header+'\n'.join(f'{label}: {text}' for label,text in zip(labels,texts))
            rendered=engine._prepare(None,prompt,labels)
            start=time.perf_counter()
            if args.serving:
                results,shared=engine.score_request(None,[field],c['state'],rotations=1)
                result=results[0];meta={**shared['questions'][0]['rotations'][0],**{k:v for k,v in shared.items() if k!='questions'}}
            else:
                result,meta=engine.score_compiled(None,prompt,labels,choices)
            calibrated=cal.calibrate_result(result,'choice',len(options),image=False)
            record={'path':'official_serving_shared_prefix' if args.serving else 'official_uncached_full_forward','id':c['id'],'offset':offset,'options':order,'gold':c['gold'],'rotations':1,'result':calibrated.model_dump(),'raw_result':result.model_dump(),'metadata':meta,'wall_seconds':time.perf_counter()-start,'hidden':capture['hidden'],'token_ids':rendered[0][0].tolist(),'prompt':prompt,'input_sha256':hashlib.sha256(bytes(json.dumps(rendered[0][0].tolist()),'utf8')).hexdigest()}
            records.append(record);output.write_text(json.dumps({'model_lock_sha256':hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest(),'records':records},indent=2)+'\n')
            print(c['id'],offset,calibrated.value,calibrated.scores,meta['input_tokens'],flush=True)
if __name__=='__main__':main()
