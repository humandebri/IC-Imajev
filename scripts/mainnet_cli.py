"""Pinned mainnet calls through icp CLI; offline Candid, resumable receipts, no key export."""
import hashlib,json,os,pathlib,re,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
TARGET='xis3j-paaaa-aaaai-axumq-cai'
IDENTITY='llm-wiki-mainnet'
OWNER='r75h6-lqd7b-5jack-at55d-vvti2-lg5qy-ly73a-5ezve-odnkc-kagu3-nae'
MODULE='5e2a0b101d4dde0ff503a8c7d399efc17e7db7618dc2c3aa6eddee49a771b3a9'
DID=ROOT/'canisters/inference/paid-inference.did'
CODEC=ROOT/'target/release/examples/mainnet_candid'
SAFE_REPLAY={'upload_chunk','hash_pack','warm_weights','prepare_fixed_prefix_state'}
def atomic_json(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)
class MainnetCLI:
    def __init__(self,directory,codec=None,floor_cycles=250_000_000_000):
        self.directory=pathlib.Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.codec=pathlib.Path(codec or CODEC);self.floor_cycles=int(floor_cycles)
        if not self.codec.is_file():raise RuntimeError(f'Missing offline codec: {self.codec}')
    def _codec(self,request):
        p=subprocess.run([str(self.codec)],input=json.dumps(request),text=True,capture_output=True,cwd=ROOT,check=False)
        if p.returncode:raise RuntimeError(f'Offline codec failed: {p.stderr}')
        return json.loads(p.stdout)
    @staticmethod
    def unwrap(value):
        if isinstance(value,dict) and 'Err' in value:raise RuntimeError(f'Canister rejected: {value["Err"]}')
        return value['Ok'] if isinstance(value,dict) and 'Ok' in value else value
    def status(self):
        p=subprocess.run(['icp','canister','status',TARGET,'--network','ic','--identity',IDENTITY,'--json'],text=True,capture_output=True,cwd=ROOT,timeout=90)
        if p.returncode:raise RuntimeError(f'Mainnet status failed: {p.stderr}')
        s=json.loads(p.stdout)
        if s['id']!=TARGET or s['module_hash'].removeprefix('0x')!=MODULE:raise RuntimeError('Target/module mismatch')
        if s['settings']['controllers']!=[OWNER]:raise RuntimeError('Controller changed')
        if int(s['settings']['freezing_threshold'].replace('_',''))!=1_209_600:raise RuntimeError('Freezing period differs from authorized 14 days')
        if int(s['settings']['compute_allocation'].replace('_',''))!=0:raise RuntimeError('Unexpected compute reservation')
        if int(s['settings']['memory_allocation'].replace('_',''))!=0:raise RuntimeError('Unexpected memory reservation')
        atomic_json(self.directory/'latest-status.json',s)
        return s
    def guard(self,growth_bytes=0):
        s=self.status();cycles=int(s['cycles'].replace('_',''));daily=int(s['idle_cycles_burned_per_day'].replace('_',''))
        freeze=int(s['settings']['freezing_threshold'].replace('_',''));existing=daily*freeze/86400
        # Empirical target-subnet rate supersedes the earlier official-table estimate.
        rate=320_000  # conservative round-up of pilot's observed 317,500 cycles/GiB/s
        projected=max(0,int(growth_bytes))*rate*freeze/(2**30)
        needed=int(existing+projected+self.floor_cycles)
        atomic_json(self.directory/'latest-budget-guard.json',{'available_cycles':cycles,'current_freezing_estimate':int(existing),'projected_growth_freezing_cycles':int(projected),'operating_floor':self.floor_cycles,'required_cycles':needed,'passed':cycles>=needed})
        if cycles<needed:raise RuntimeError(f'Budget guard: available {cycles} < freeze+growth+floor {needed}; do not mint/topup automatically')
        return s
    def call(self,method,args=None,query=False,key=None,idempotent=False,did=None):
        args=[] if args is None else args
        if idempotent and not query and method not in SAFE_REPLAY:raise ValueError(f'Not an approved replay-safe method: {method}')
        if key is None:key=method+'-'+hashlib.sha256(json.dumps(args,sort_keys=True).encode()).hexdigest()[:16]
        if query:key+=f'-query-{time.time_ns()}'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',key):raise ValueError('Unsafe receipt key')
        dest=self.directory/'calls'/key;dest.mkdir(parents=True,exist_ok=True)
        arg=dest/'args.bin';info=self._codec({'op':'encode','method':method,'args':args,'output':str(arg)})
        if info['bytes']>=1_990_000:raise RuntimeError('Candid request exceeds safe mainnet ingress envelope')
        digest=hashlib.sha256(arg.read_bytes()).hexdigest()
        meta={'target':TARGET,'network':'ic','identity':IDENTITY,'module_hash':MODULE,'method':method,'query':query,'args_sha256':digest,'request_bytes':info['bytes'],'idempotent':idempotent}
        previous=dest/'request.json'
        if previous.exists() and json.loads(previous.read_text())!=meta:raise RuntimeError('Receipt request changed')
        atomic_json(previous,meta)
        reply=dest/'reply.hex'
        if reply.exists():
            result=self._codec({'op':'decode','method':method,'hex':reply.read_text()})['result']
            atomic_json(dest/'decoded.json',result);return result
        sent=dest/'sent.json'
        if sent.exists() and not query and not idempotent:raise RuntimeError(f'Unresolved non-idempotent update at {dest}; do not blindly replay')
        argv=['icp','canister','call',TARGET,method,'--network','ic','--identity',IDENTITY,'--candid',str(did or DID),'--args-file',str(arg),'--args-format','bin','--output','hex']
        if query:argv+=['--query']
        attempts=3 if query or idempotent else 1
        for attempt in range(attempts):
            atomic_json(sent,{'started_unix':time.time(),'attempt':attempt+1,'argv':argv})
            started=time.monotonic()
            try:p=subprocess.run(argv,text=True,capture_output=True,cwd=ROOT,timeout=900)
            except subprocess.TimeoutExpired as e:
                atomic_json(dest/f'error-{attempt}.json',{'error':'CLI timeout, update outcome may be unknown','stdout':str(e.stdout),'stderr':str(e.stderr)})
                if attempt+1>=attempts:raise RuntimeError(f'Unknown outcome: {method} at {dest}') from e
                time.sleep(2**attempt);continue
            (dest/f'stderr-{attempt}.txt').write_text(p.stderr)
            if p.returncode:
                atomic_json(dest/f'error-{attempt}.json',{'returncode':p.returncode,'stderr':p.stderr,'stdout':p.stdout})
                transient=any(x in p.stderr.lower() for x in ['sysTransient'.lower(),'temporarily','timeout','timed out','connection reset','connection closed','transport error','http error: status 429','http error: status 503','http error: status 502'])
                if transient and attempt+1<attempts:time.sleep(2**attempt);continue
                raise RuntimeError(f'{method} CLI failed ({p.returncode}): {p.stderr}')
            raw=p.stdout.strip()
            if not re.fullmatch(r'(0x)?[0-9a-fA-F]+',raw):raise RuntimeError(f'Unexpected CLI hex reply at {dest}')
            reply.write_text(raw+'\n')
            result=self._codec({'op':'decode','method':method,'hex':raw})['result']
            atomic_json(dest/'decoded.json',result)
            atomic_json(dest/'completed.json',{'wall_seconds':time.monotonic()-started,'reply_bytes':len(raw.removeprefix('0x'))//2,'completed_unix':time.time()})
            return result
        raise RuntimeError('Retry loop exhausted')
