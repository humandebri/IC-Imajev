#!/usr/bin/env python3
"""Mainnet pinned model upload through icp CLI only. No funding or settings changes."""
import argparse,concurrent.futures,hashlib,json,pathlib,sys,time
from mainnet_cli import MainnetCLI,ROOT,atomic_json
PACK_SHA='364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84'
MODEL='7ef38ecb70f4afa4e92bad2fe9f0448699ee45041688b1110819143f41601330'
CHUNK=1_800_000

def validate(pack,manifest_path,directory):
    m=json.loads(manifest_path.read_text());size=pack.stat().st_size
    if m['model']!=MODEL or m['pack_hash']!=PACK_SHA or m['bytes']!=4_702_451_200 or size!=m['bytes']:raise RuntimeError('Model/pack length identity mismatch')
    if hashlib.sha256((ROOT/'MODEL_LOCK.json').read_bytes()).hexdigest()!=MODEL:raise RuntimeError('Model lock changed')
    h=hashlib.sha256()
    with pack.open('rb') as f:
        while data:=f.read(16*1024*1024):h.update(data)
    if h.hexdigest()!=PACK_SHA:raise RuntimeError('Full pack SHA256 mismatch')
    proof={'model':MODEL,'pack_hash':PACK_SHA,'bytes':size,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'tensor_count':len(m['tensors']),'chunks':(size+CHUNK-1)//CHUNK,'chunk_bytes':CHUNK,'pack_sha256_verified':True}
    atomic_json(directory/'pack-provenance.json',proof);return m,proof

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--directory',default='artifacts/mainnet-deploy-20261007');ap.add_argument('--pack',default='checkpoints/full-int8.pack');ap.add_argument('--manifest',default='checkpoints/full-int8.manifest.json');ap.add_argument('--concurrency',type=int,default=8);ap.add_argument('--pilot-chunks',type=int,default=0);ap.add_argument('--codec');a=ap.parse_args()
    if not 1<=a.concurrency<=16:raise ValueError('Concurrency must be 1..16')
    directory=ROOT/a.directory;directory.mkdir(parents=True,exist_ok=True);pack=ROOT/a.pack;mp=ROOT/a.manifest
    m,proof=validate(pack,mp,directory);print(json.dumps({'stage':'offline-pack-verified',**proof}),flush=True)
    if not a.execute:return
    cli=MainnetCLI(directory/'upload',codec=a.codec)
    before=cli.status()
    if int(before['settings']['wasm_memory_limit'].replace('_',''))!=2**32:raise RuntimeError('4GiB Wasm memory limit required')
    status=cli.call('pack_status',query=True)
    if not status['model']:
        cli.guard(growth_bytes=m['bytes'])
        cli.unwrap(cli.call('prepare',[mp.read_text()],key='prepare-manifest'))
        status=cli.call('pack_status',query=True)
    if status['model']!=MODEL or status['pack_hash']!=PACK_SHA or status['bytes']!=m['bytes']:raise RuntimeError('Existing pack differs; never reset it automatically')
    if status['ready']:
        if status['received']!=status['hashed'] or status['hashed']!=m['bytes']:raise RuntimeError('Invalid ready status')
        atomic_json(directory/'upload-verified.json',{'pack':status,'status':cli.status(),'already_ready':True});print(json.dumps({'stage':'already-ready'}),flush=True);return
    def pack_state(min_received=0,min_chunks=0,min_hashed=0):
        # Queries may reach a lagging replica immediately after certified updates.
        for attempt in range(10):
            value=cli.call('pack_status',query=True)
            if value['model']!=MODEL or value['pack_hash']!=PACK_SHA or value['bytes']!=m['bytes']:raise RuntimeError('Query returned different pack identity')
            if value['received']>=min_received and len(value['chunks'])>=min_chunks and value['hashed']>=min_hashed:return value
            if attempt<9:time.sleep(min(1+attempt,3))
        raise RuntimeError('Pack query remained behind known committed updates; stop rather than pay for blind reuploads')
    completed=set(status['chunks']);known_chunks=len(completed);known_received=status['received'];offsets=[o for o in range(0,m['bytes'],CHUNK) if o not in completed]
    if a.pilot_chunks:offsets=offsets[:a.pilot_chunks]
    data_dir=directory/'upload'/'payloads';data_dir.mkdir(parents=True,exist_ok=True)
    newly_uploaded=0;started=time.monotonic();history=[]
    def upload(offset):
        with pack.open('rb') as f:f.seek(offset);data=f.read(min(CHUNK,m['bytes']-offset))
        if len(data)!=min(CHUNK,m['bytes']-offset):raise RuntimeError('Pack truncated')
        path=data_dir/f'{offset:012d}.bin';path.write_bytes(data)
        sha=hashlib.sha256(data).hexdigest()
        try:
            value=cli.unwrap(cli.call('upload_chunk',[offset,{'file':str(path)},{'hex':sha}],key=f'chunk-{offset:012d}',idempotent=True))
            return {'offset':offset,'bytes':len(data),'sha256':sha,'received':value}
        finally:path.unlink(missing_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for index in range(0,len(offsets),32):
            cli.guard()
            batch=offsets[index:index+32]
            futures=[pool.submit(upload,o) for o in batch]
            for f in concurrent.futures.as_completed(futures):
                row=f.result();newly_uploaded+=1;history.append(row);completed.add(row['offset']);known_received=max(known_received,row['received'])
            known_chunks=len(completed)
            status=pack_state(min_received=known_received,min_chunks=known_chunks);now=cli.status()
            if status['received']<0 or status['received']>m['bytes']:raise RuntimeError('Bad upload progress')
            atomic_json(directory/'upload-progress.json',{'received':status['received'],'hashed':status['hashed'],'ready':status['ready'],'completed_chunks':len(status['chunks']),'new_chunks_this_run':newly_uploaded,'canister_cycles':now['cycles'],'initial_canister_cycles':before['cycles'],'wall_seconds':time.monotonic()-started})
            with (directory/'upload-chunks.jsonl').open('a') as log:
                for row in history:log.write(json.dumps(row)+'\n')
            history=[]
            print(json.dumps({'stage':'upload','received':status['received'],'total':m['bytes'],'chunks':len(status['chunks']),'cycles':now['cycles'],'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    status=pack_state(min_received=known_received,min_chunks=known_chunks)
    if a.pilot_chunks and status['received']<m['bytes']:
        after=cli.status();atomic_json(directory/'pilot.json',{'before':before,'after':after,'pack':status,'new_chunks':newly_uploaded,'cycles_delta':int(before['cycles'].replace('_',''))-int(after['cycles'].replace('_','')),'scope':'Preparation growth and this pilot upload; not full deployment cost'})
        print(json.dumps({'stage':'pilot-complete','chunks_this_run':newly_uploaded,'received':status['received']}),flush=True);return
    if status['received']!=m['bytes'] or len(status['chunks'])!=proof['chunks']:raise RuntimeError('Incomplete upload; refusing to seal')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        while not status['ready']:
            cli.guard(growth_bytes=16*1024*1024);old=status['hashed']
            count=min(a.concurrency,(m['bytes']-old+7_999_999)//8_000_000)
            def hash_batch(i):
                return cli.unwrap(cli.call('hash_pack',[8_000_000],key=f'hash-{old:012d}-{i:02d}',idempotent=True))
            # Each owner update atomically advances the same SHA state; no input
            # hash state is client supplied. Exact max_bytes calls are replay safe.
            progress=list(pool.map(hash_batch,range(count)))
            largest=max(x[0] for x in progress)
            if largest<=old or largest>m['bytes']:raise RuntimeError('Hash progress invalid')
            status=pack_state(min_received=m['bytes'],min_chunks=proof['chunks'],min_hashed=largest)
            print(json.dumps({'stage':'hash','hashed':status['hashed'],'total':m['bytes'],'ready':status['ready']}),flush=True)
    if status['model']!=MODEL or status['pack_hash']!=PACK_SHA or status['received']!=status['hashed'] or status['hashed']!=m['bytes']:raise RuntimeError('Final sealedpack identity mismatch')
    atomic_json(directory/'upload-verified.json',{'pack':status,'status':cli.status(),'new_chunks_this_run':newly_uploaded,'wall_seconds':time.monotonic()-started,'scope':'Mainnet upload and full SHA256 verification; runtime automatically sets ready after successful full hash'})
    print(json.dumps({'stage':'pack-ready','bytes':status['bytes'],'pack_hash':status['pack_hash']}),flush=True)
if __name__=='__main__':main()
