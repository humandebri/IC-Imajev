"""Raw Candid calls with request/reply evidence, including cycles-attaching relay."""
import json,subprocess,time
from pathlib import Path
from canister_api_names import METHOD_NAMES
ROOT=Path(__file__).resolve().parents[1]
def evidence_path(path):
 path=Path(path)
 try:return str(path.relative_to(ROOT))
 except ValueError:return str(path)
class PaidTransport:
 def __init__(self,directory,target,helper=None,call_helper=None,*,url=None,identity_pem=None):
  self.d=Path(directory)
  if not self.d.is_absolute():self.d=ROOT/self.d
  self.d.mkdir(parents=True,exist_ok=True);self.target=target;self.counter=0;self.helper=Path(helper) if helper is not None else ROOT/'target/debug/examples/paid_update_args'
  self.call_helper=Path(call_helper) if call_helper is not None else ROOT/'target/debug/examples/paid_raw_call'
  if not self.call_helper.is_file():raise RuntimeError('Run cargo build -p imajev-client --example paid_raw_call first')
  self.call_options=[]
  if url is not None:self.call_options+=['--url',str(url)]
  if identity_pem is not None:self.call_options+=['--identity-pem',str(identity_pem)]
  self.did=self.d/'wire.did';self.did.write_text('service:{getInferenceQuote:()->(blob)query;runPaidInference:()->(blob);configurePaidInference:()->(blob);getInferenceReceipt:()->(blob)query;runPaidInferenceWorker:()->(blob);retryInferenceRefund:()->(blob);getPaidInferenceDebug:()->(blob)query;getPaidInferenceConfig:()->(blob)query;setPaidInferenceFault:()->();forward:()->(blob);balance:()->(nat)query;}')
 def args(self,kind,value,prefix):
  j=self.d/(prefix+'.input.json');j.write_text(json.dumps(value)+'\n');b=self.d/(prefix+'.args.bin');subprocess.run([str(self.helper),'args',kind,str(j),str(b)],check=True);return b
 def decode(self,kind,hexpath):return json.loads(subprocess.check_output([str(self.helper),'decode',kind,str(hexpath)],text=True))
 def call(self,kind,value=None,relay=None,cycles=0):
  self.counter+=1;prefix=f'{self.counter:04d}-{kind}';arg=self.args(kind,value,prefix);target=self.target;method=METHOD_NAMES.get(kind,kind)
  if relay:
   arg=self.args('forward',dict(target=target,method=method,args=str(arg),cycles=cycles),prefix+'-relay');target=relay;method='forward'
  started=time.monotonic();cmd=[str(self.call_helper),*self.call_options,target,method,str(arg)]
  raw=subprocess.check_output(cmd,cwd=ROOT,text=True);hexpath=self.d/(prefix+'.reply.hex');hexpath.write_text(raw);elapsed=time.monotonic()-started
  if relay:
   forward=self.decode('forward',hexpath)
   if 'Err'in forward['response']:result=dict(transport_error=forward['response']['Err'])
   else:
    inner=self.d/(prefix+'.inner.hex');inner.write_text(bytes(forward['response']['Ok']).hex());result=self.decode(kind,inner)
  else:forward=None;result=self.decode(kind,hexpath)
  row=dict(kind=kind,request=value,relay=relay,attached_cycles=cycles,result=result,forward=forward,seconds=elapsed,args_path=evidence_path(arg),reply_path=evidence_path(hexpath),request_bytes=arg.stat().st_size,reply_bytes=len(raw.strip().removeprefix('0x'))//2)
  (self.d/(prefix+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
