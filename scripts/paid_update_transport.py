"""Raw Candid calls with request/reply evidence, including cycles-attaching relay."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PaidTransport:
 def __init__(self,directory,target):
  self.d=Path(directory);self.d.mkdir(parents=True,exist_ok=True);self.target=target;self.counter=0;self.helper=ROOT/'artifacts/paid-update-v1/tools/args'
  self.did=self.d/'wire.did';self.did.write_text('service:{quote:()->(blob)query;infer:()->(blob);configure_paid:()->(blob);inference_status:()->(blob)query;inference_step:()->(blob);retry_inference_refund:()->(blob);paid_debug:()->(blob)query;paid_config:()->(blob)query;paid_fault:()->();forward:()->(blob);balance:()->(nat)query;}')
 def args(self,kind,value,prefix):
  j=self.d/(prefix+'.input.json');j.write_text(json.dumps(value)+'\n');b=self.d/(prefix+'.args.bin');subprocess.run([str(self.helper),'args',kind,str(j),str(b)],check=True);return b
 def decode(self,kind,hexpath):return json.loads(subprocess.check_output([str(self.helper),'decode',kind,str(hexpath)],text=True))
 def call(self,kind,value=None,relay=None,cycles=0):
  self.counter+=1;prefix=f'{self.counter:04d}-{kind}';arg=self.args(kind,value,prefix);target=self.target;method=kind
  if relay:
   arg=self.args('forward',dict(target=target,method=kind,args=str(arg),cycles=cycles),prefix+'-relay');target=relay;method='forward'
  started=time.monotonic();cmd=['icp','canister','call',target,method,'--network','local','--identity','imajev-local','--candid',str(self.did),'--args-file',str(arg),'--args-format','bin','--output','hex']
  raw=subprocess.check_output(cmd,cwd=ROOT,text=True);hexpath=self.d/(prefix+'.reply.hex');hexpath.write_text(raw);elapsed=time.monotonic()-started
  if relay:
   forward=self.decode('forward',hexpath)
   if 'Err'in forward['response']:result=dict(transport_error=forward['response']['Err'])
   else:
    inner=self.d/(prefix+'.inner.hex');inner.write_text(bytes(forward['response']['Ok']).hex());result=self.decode(kind,inner)
  else:forward=None;result=self.decode(kind,hexpath)
  row=dict(kind=kind,request=value,relay=relay,attached_cycles=cycles,result=result,forward=forward,seconds=elapsed,args_path=str(arg.relative_to(ROOT)),reply_path=str(hexpath.relative_to(ROOT)),request_bytes=arg.stat().st_size,reply_bytes=len(raw.strip().removeprefix('0x'))//2)
  (self.d/(prefix+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
