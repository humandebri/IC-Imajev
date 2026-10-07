#!/usr/bin/env python3
"""Freeze and download the reconstructed local-only baseline before paid proofs."""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport
from prefix_inference import verify_module
TARGET='4caro-hl777-77775-aaaba-cai';BASE='6052cc94ffb6285edfc3343c3edf5ae8de24d048188b282a7f88451beba0e931'
D=ROOT/'artifacts/local-goal-recovery-v1'
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--backup', type=Path, required=True, help='new snapshot output directory; never chosen automatically');args=parser.parse_args();BACKUP=args.backup.expanduser().resolve()
 if BACKUP.exists() or not BACKUP.parent.is_dir():parser.error('--backup must be a new directory with an existing parent')
 assert not (D/'baseline-ready.json').exists()
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());old=json.loads((ROOT/'artifacts/paid-owned-profile-off-v1/proof/before.json').read_text());events=[]
 def command(*args):
  started=time.monotonic();p=subprocess.run(['icp','canister',*args,'--network','local','--identity','imajev-local'],cwd=ROOT,text=True,capture_output=True);events.append(dict(args=list(args),code=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-started));(D/'checkpoint-operations.json').write_text(json.dumps(events,indent=2)+'\n');assert p.returncode==0,p.stderr;return p.stdout.strip()
 def state(name):
  t=Transport(m['model'],'http://localhost:8001/',TARGET,str(ROOT/'artifacts/imajev-local.pem'),D/name,m['pack_hash'],bridge_binary=str(ROOT/'artifacts/query-packing-v3/build/imajev-client'))
  try:verify_module(t,BASE);return dict(module=BASE,cache=t.command(dict(op='weight_cache_status'))['ok']['cache'],pack=t.command(dict(op='pack_status'))['ok'])
  finally:t.close()
 before=state('checkpoint-before');assert before['cache']==old['cache']
 for key in ['bytes','hashed','model','pack_hash','ready','received']:assert before['pack'][key]==old['pack'][key]
 (D/'baseline-state.json').write_text(json.dumps(before,indent=2)+'\n');print('baseline cache and critical pack fields match frozen original',flush=True)
 command('stop',TARGET);snapshot=command('snapshot','create',TARGET,'--quiet');(D/'durable-snapshot.json').write_text(json.dumps(dict(id=snapshot,target=TARGET,backup=str(BACKUP)),indent=2)+'\n');command('start',TARGET)
 command('snapshot','download',TARGET,snapshot,'--output',str(BACKUP));print('snapshot download terminal succeeded',flush=True)
 files=sorted(p for p in BACKUP.rglob('*')if p.is_file());assert files
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb')as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  return h.hexdigest()
 hashes={str(p.relative_to(BACKUP)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in files};(D/'durable-files.json').write_text(json.dumps(dict(root=str(BACKUP),files=hashes),indent=2)+'\n')
 after=state('checkpoint-after');assert after==before
 (D/'baseline-ready.json').write_text(json.dumps(dict(complete=True,target=TARGET,module=BASE,cache_equal_to_original=True,critical_pack_fields_equal_to_original=True,snapshot=snapshot,download_terminal=True,backup=str(BACKUP),download_files_hashed=True,state_unchanged_after_download=True),indent=2)+'\n');print('durable baseline ready',flush=True)
if __name__=='__main__':main()
