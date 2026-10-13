#!/usr/bin/env python3
"""Acquire immutable upstream files; inventory safetensors headers before full weights."""
import argparse, hashlib, json, pathlib, struct, urllib.request, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER = 'c9e5f132465da85d31735ec502d5557982671a7d'
BASE = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SERVER = 'a0134749e0900189c129cd6bb5000969f3b64bb5'
def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.part')
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as src, temp.open('wb') as dst:
                    while block := src.read(8*1024*1024): dst.write(block)
                temp.replace(path); break
            except Exception:
                if attempt == 3: raise
                time.sleep(2**attempt)
    return {'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size, 'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
def header(url):
    def read(a,b):
        req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
        with urllib.request.urlopen(req,timeout=120) as r:
            if r.status != 206: raise ValueError('Server did not honor Range')
            return r.read()
    n=struct.unpack('<Q',read(0,7))[0]
    if n>16*1024*1024: raise ValueError('oversized header')
    return json.loads(read(8,7+n)),8+n
if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--weights',action='store_true');args=ap.parse_args()
    lock={'schema':1,'adapter_revision':ADAPTER,'base_revision':BASE,'server_commit':SERVER,'rotations':1,'text_only':True,'components':{},'tensor_inventory':{}}
    for role,repo,rev,names in [
        ('base','Qwen/Qwen3.5-4B',BASE,['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','video_preprocessor_config.json','chat_template.jinja','merges.txt','vocab.json','LICENSE']),
        ('adapter','mohit67890/imajev-4b',ADAPTER,['adapter_config.json','decision_readout.json','calibration.json','ckpt.json','RELEASE-SPEC.md','SHA256SUMS','README.md','mlx/adapter_config.json','mlx/decision_readout.json'])]:
        records={}
        for name in names:
            print('metadata',role,name,flush=True)
            url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
            records[name]=fetch(url,ROOT/'checkpoints'/role/name)
        tensors = (list(set(json.loads((ROOT/'checkpoints/base/model.safetensors.index.json').read_text())['weight_map'].values())) if role=='base' else ['adapter_model.safetensors','decision_readout.safetensors','mlx/adapters.safetensors','mlx/decision_readout.safetensors'])
        for name in sorted(tensors):
            url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
            print('header',name,flush=True)
            h,offset=header(url)
            size=offset+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
            lock['tensor_inventory'][f'{role}/{name}']={'file_bytes':size,'header_bytes':offset,'tensors':h}
            if args.weights or 'readout' in name:
                print('download',name,size,flush=True)
                records[name]=fetch(url,ROOT/'checkpoints'/role/name)
                if records[name]['bytes']!=size: raise ValueError('size mismatch')
        lock['components'][role]={'repo':repo,'revision':rev,'files':records}
    server=ROOT/f'vendor/imajev-{SERVER}'
    lock['official_source_sha256']={str(p.relative_to(server)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(server.rglob('*.py'))}
    path=ROOT/'MODEL_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('Existing lock differs; refusing to replace immutable model bundle')
    if not path.exists():path.write_text(json.dumps(lock,indent=2)+'\n')
