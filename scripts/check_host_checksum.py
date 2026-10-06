#!/usr/bin/env python3
"""Validate host-sealed frames against real signed local query transport."""
import argparse,hashlib,http.server,json,pathlib,struct,sys,threading,urllib.request,urllib.error
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'client'))
from transport import Transport,encode,decode,atomic
from prefix_inference import verify_module

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['canister','wasm','directory']:ap.add_argument('--'+name,required=True)
 a=ap.parse_args();d=ROOT/a.directory;d.mkdir(parents=True,exist_ok=True)
 m=json.loads((ROOT/'checkpoints/full-int8.manifest.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
 h=dict(version=3,model=m['model'],pack_hash=m['pack_hash'],input_hash='c'*64,step=0,op='bf16',tensor='',dims=[],scalars=[],encoding='bf16-block256-exact-v1')
 x=np.array([0.,-0.,1.0000001,1e-40,4.25],np.float32);request=d/'request.bin';atomic(request,encode(h,x));rows=[]
 def bridge(url):return Transport(m['model'],url,a.canister,str(ROOT/'artifacts/imajev-local.pem'),d,m['pack_hash'],frame_version=3)
 t=bridge('http://localhost:8001/')
 try:
  verify_module(t,sha((ROOT/a.wasm).read_bytes()))
  reply=d/'valid.response.bin';measurement=t.command(dict(op='step',input=str(request),output=str(reply)));rh,y=decode(reply.read_bytes())
  assert rh==dict(h,step=1)
  # bf16's rounding is identical in both transport versions.
  legacy=d/'legacy.request.bin';out=d/'legacy.response.bin';atomic(legacy,encode(dict(h,version=2),x));t.command(dict(op='step',input=str(legacy),output=str(out)));assert decode(out.read_bytes())[1].tobytes()==y.tobytes()
  rows.append(dict(check='valid_signed_reply_and_legacy_bits',measurement=measurement))
 finally:t.close()
 counters={'queries':0,'tampered':0}
 class Proxy(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def forward(self):
   body=self.rfile.read(int(self.headers.get('Content-Length',0))) if self.command=='POST' else None
   req=urllib.request.Request('http://localhost:8001'+self.path,data=body,method=self.command,headers={'Content-Type':self.headers.get('Content-Type','application/cbor')})
   try:r=urllib.request.urlopen(req,timeout=30)
   except urllib.error.HTTPError as e:r=e
   with r:payload=r.read();status=r.status;ctype=r.headers.get('Content-Type','application/cbor')
   if self.command=='POST' and self.path.endswith('/query'):
    counters['queries']+=1;pos=payload.find(b'DIDL');assert pos>=0,'Expected Candid reply inside CBOR'
    payload=payload[:pos]+b'E'+payload[pos+1:];counters['tampered']+=1
   self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
  do_GET=forward;do_POST=forward
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Proxy);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();t=bridge(f'http://127.0.0.1:{server.server_port}/')
 try:
  for name,pos in [('payload',4+struct.unpack('<I',request.read_bytes()[:4])[0]),('footer',len(request.read_bytes())-1)]:
   b=bytearray(request.read_bytes());b[pos]^=1;p=d/f'bad-{name}.bin';atomic(p,b);out=d/f'bad-{name}.response.bin'
   try:t.command(dict(op='step',input=str(p),output=str(out)))
   except RuntimeError as e:assert 'checksum' in str(e).lower();rows.append(dict(check='stored_'+name,error=str(e)))
   else:raise AssertionError('Corrupted saved state accepted')
   assert counters['queries']==0 and not out.exists()
  out=d/'tampered.response.bin'
  try:t.command(dict(op='step',input=str(request),output=str(out)))
  except RuntimeError as e:
   assert 'signature' in str(e).lower(),str(e);rows.append(dict(check='tampered_signed_response',error=str(e)))
  else:raise AssertionError('Tampered query signature accepted')
  assert counters==dict(queries=1,tampered=1) and not out.exists()
 finally:t.close();server.shutdown();server.server_close();thread.join()
 report=dict(canister=a.canister,wasm_sha256=sha((ROOT/a.wasm).read_bytes()),bridge_sha256=sha((ROOT/'target/release/imajev-client').read_bytes()),checks=rows,proxy=counters,scope='Local signed query transport and durable checksum; not replicated execution certification')
 (d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
