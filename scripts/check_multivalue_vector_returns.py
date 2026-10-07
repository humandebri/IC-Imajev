#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];CAN='4xhad-gd777-77775-aaacq-cai'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=ROOT/'artifacts/multivalue-vector-returns-v1';b=json.loads((d/'build.json').read_text())
 for p,h in b['source_hashes'].items():assert sha(ROOT/p)==h,p
 def status():return json.loads(subprocess.check_output(['icp','canister','status',CAN,'--network','local','--identity','imajev-local','--json'],text=True))
 assert status()['module_hash']=='0x'+b['module']
 check=d/'check';check.mkdir(exist_ok=False);measurements={};replies=[]
 for mode in ['query','update']:
  for name,m in b['methods'].items():
   cmd=['icp','canister','call',CAN,name if mode=='query'else name+'_update','()','--network','local','--identity','imajev-local','--candid',str(d/f'{mode}.did'),'--output','hex']
   if mode=='query':cmd+=['--query']
   raw=subprocess.check_output(cmd,text=True);p=check/f'{mode}-{name}.hex';p.write_text(raw)
   data=bytes.fromhex(raw.strip().removeprefix('0x'));assert len(data)==24 and data[:8]==b'DIDL\0\2\x78\x78'
   count=int.from_bytes(data[8:16],'little');checksum=int.from_bytes(data[16:24],'little');assert checksum==m['expected_checksum']
   measurements[mode+'/'+name]=count;replies.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),instructions=count,checksum=checksum))
   print(json.dumps(dict(mode=mode,name=name,instructions=count,checksum=checksum)),flush=True)
 assert status()['module_hash']=='0x'+b['module']
 assert all(measurements['query/'+n]==measurements['update/'+n]for n in b['methods'])
 extras={mode:{str(n):(measurements[f'{mode}/helper{n}']-measurements[f'{mode}/inline{n}'])/1000 for n in [16,64,128,256]}for mode in ['query','update']}
 files=[Path(__file__),d/'build.json']+[ROOT/p for p in b['source_hashes']]+[ROOT/r['path']for r in replies]
 r=dict(complete=True,module=b['module'],measurements=measurements,helper_extra_per_call=extras,replies=replies,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in files},scope='Multi-value vector return ABI and matched compound cost; no inference performance or fidelity claim.')
 (d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 with zipfile.ZipFile(d/'evidence.zip','x',zipfile.ZIP_DEFLATED)as z:
  for p in files+[d/'report.json']:z.write(p,str(p.relative_to(ROOT)))
 print(json.dumps(dict(helper_extra_per_call=extras)),flush=True)
if __name__=='__main__':main()
