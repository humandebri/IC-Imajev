#!/usr/bin/env python3
"""Re-audit completed saved MLP carries against raw paid replies and freeze evidence."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/terminal-mlp-reference-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest()
def main():
 extraction=json.loads((D/'report.json').read_text());r=json.loads((D/'reconstruction/report.json').read_text())
 paid_dir=ROOT/'artifacts/paid-finite-simd-v1';paid=json.loads((paid_dir/'proof/report.json').read_text());summary=json.loads((paid_dir/'summary.json').read_text())
 assert extraction['complete'] and extraction['extraction_only'] and r['complete'] and not r['pilot']
 assert paid['complete'] and paid['baseline_restored'] and summary['complete'] and summary['module']==paid['candidate']
 hashes={}
 for manifest in [extraction['source_hashes'],r['source_hashes'],r['reference_hashes'],summary['workflow_hashes'],summary['reference_hashes']]:
  for p,h in manifest.items():assert sha(ROOT/p)==h,p;hashes[p]=h
 manifest=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_text());identity=json.loads((D/'pack-identity.json').read_text())
 assert identity['pack_hash']==manifest['pack_hash'] and identity['bytes']==(ROOT/'checkpoints/full-int8.pack').stat().st_size
 assert sha(ROOT/'MODEL_LOCK.json')==manifest['model']
 tensors={t['name']:t for t in manifest['tensors']}
 with (ROOT/'checkpoints/full-int8.pack').open('rb')as f:
  for name,h in r['weight_hashes'].items():
   t=tensors[name];f.seek(t['offset']);raw=f.read(t['bytes']);assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==h,name
 helper=ROOT/'artifacts/paid-update-v1/tools/args';debugs=[];raw_files=[]
 for p in sorted((paid_dir/'proof').rglob('*.json')):
  c=json.loads(p.read_text())
  if isinstance(c,dict) and c.get('kind')=='paid_debug' and 'reply_path' in c:
   assert not c['relay'];reply=ROOT/c['reply_path'];actual=json.loads(subprocess.check_output([str(helper),'decode','paid_debug',str(reply)],text=True));assert actual==c['result']
   debugs.append(actual);raw_files.extend([p,reply,ROOT/c['args_path']])
 assert [(v['layer'],v['case'])for v in r['results']]==[(layer,case)for layer in [26,30]for case in ['617','620','653']]
 carry={(v['case'],v['layer']):v for v in extraction['cases']};latest={v['case']:v for v in paid['results']};checks=[]
 for v in r['results']:
  c=carry[v['case'],v['layer']];assert c['header']['model']==manifest['model'] and c['header']['pack_hash']==manifest['pack_hash']
  for p,h in [(c['source'],c['source_sha256']),(c['carry'],c['carry_sha256']),(v['output'],v['output_sha256'])]:assert sha(ROOT/p)==h,p;hashes[p]=h
  hidden=np.load(ROOT/v['output'],allow_pickle=False);assert hidden.shape==(c['n'],2560) and hidden.dtype==np.dtype('<f4') and np.isfinite(hidden).all()
  item=latest[v['case']];debug=item['debug'];assert debug in debugs and len(debug['hidden_hashes'])==32
  offset=item['quote']['prefix_tokens']-c['prefix'];assert offset==v['paid_suffix_offset']>=0
  assert digest(hidden[offset:])==v['hidden_hash']==debug['hidden_hashes'][v['layer']]
  if v['layer']==26:
   expected=np.load(ROOT/f"artifacts/boomdao-current-v1/{v['case']}-r1/queries/layer-26.npy",allow_pickle=False)[c['prefix']:]
   assert np.array_equal(hidden.view('<u4'),expected.view('<u4')) and v['historical_reference_bit_equal']
  else:assert not v['historical_reference_bit_equal']
  checks.append(dict(case=v['case'],layer=v['layer'],values=int(hidden.size),reference_kind='saved historical hidden'if v['layer']==26 else'independent continuation of saved exact carry',bit_equal=True,paid_suffix_offset=offset,hidden_hash=v['hidden_hash']))
 # Recheck the other 31 historical hidden arrays against the same raw paid debug.
 for case,item in latest.items():
  for layer in range(32):
   if layer==30:continue
   p=ROOT/f'artifacts/boomdao-current-v1/{case}-r1/queries/layer-{layer:02d}.npy';a=np.load(p,allow_pickle=False)
   if layer<31:a=a[item['quote']['prefix_tokens']:]
   assert digest(a)==item['debug']['hidden_hashes'][layer],(case,layer)
 files=[Path(__file__),ROOT/'MODEL_LOCK.json',D/'pack-identity.json',D/'report.json',D/'reconstruction/report.json',D/'reconstruction/progress.json',paid_dir/'summary.json',helper]+raw_files
 hashes.update({str(p.relative_to(ROOT)):sha(p)for p in files})
 result=dict(complete=True,scope='Independent host continuation of saved exact carries, validated against historical layer26; all 32 hidden layers match saved raw paid debug. Layer30 is reconstructed, not a historical exported hidden array. No new instruction-count measurement.',module=paid['candidate'],checks=checks,all_32_hidden_verified=True,saved_paid_debug_redecoded=True,pack_identity=identity,weight_hashes=r['weight_hashes'],workflow_hashes=hashes,instructions=summary['cases'],all_targets_met=summary['all_targets_met'])
 (D/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with zipfile.ZipFile(D/'frozen-workflow.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in sorted(set(hashes)|{str((D/'summary.json').relative_to(ROOT))}):z.write(ROOT/p,p)
 print(json.dumps(dict(complete=True,all_32_hidden_verified=True,checks=checks,all_targets_met=result['all_targets_met']),indent=2))
if __name__=='__main__':main()
