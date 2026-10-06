import fs from 'node:fs';import crypto from 'node:crypto';
const [dir]=process.argv.slice(2),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const fnv=b=>{let h=14695981039346656037n;for(const x of b)h=BigInt.asUintN(64,(h^BigInt(x))*1099511628211n);return h.toString(16).padStart(16,'0');};
const wasm=fs.readFileSync(`${dir}/diagnostic.wasm`);
const {instance}=await WebAssembly.instantiate(wasm,{ic0:{performance_counter:()=>0n,msg_arg_data_size:()=>0,msg_arg_data_copy:()=>{},msg_reply_data_append:()=>{},msg_reply:()=>{}}});const e=instance.exports;
const output=()=>Buffer.from(new Uint8Array(e.memory.buffer,e.output_ptr(),e.output_len()));
const specs=[];
for(const cols of [1,3,4,5,15,16,17,1024,2624,4096,5248,16384])for(const pattern of [0,1,2,3,4,5,6,7,8])specs.push({tokens:3,cols,pattern});
for(const cols of [1024,2624,4096,5248])for(const tokens of [1,7,32,65,128])specs.push({tokens,cols,pattern:0});
const rows=[];
for(const s of specs){
 e.setup(s.tokens,s.cols,s.pattern);const input=Float32Array.from(new Float32Array(e.memory.buffer,e.input_ptr(),s.tokens*s.cols));
 const valid=e.run(0),base=output();if(e.run(1)!==valid||!base.equals(output()))throw new Error(`bit mismatch ${JSON.stringify(s)}`);
 const ref=Buffer.alloc(s.tokens*4+input.length);for(let t=0;t<s.tokens;t++)ref.writeFloatLE(1,t*4);let scalarValid=true;
 for(let t=0;t<s.tokens;t++){
  const values=input.slice(t*s.cols,(t+1)*s.cols);let peak=0;for(const v of values){if(!Number.isFinite(v)){scalarValid=false;break;}peak=Math.max(peak,Math.abs(v));}if(!scalarValid)break;
  const scale=peak===0?1:Math.max(Math.fround(peak/127),Math.fround(2**-126));ref.writeFloatLE(scale,t*4);
  for(let c=0;c<s.cols;c++){const v=Math.fround(values[c]/scale),a=Math.abs(v),whole=Math.trunc(a);const q=Math.min(127,whole+(a-whole>=.5?1:0))*(v<0?-1:1);ref[s.tokens*4+t*s.cols+c]=q&255;}
 }
 if(scalarValid!==!!valid||!base.equals(ref))throw new Error(`scalar mismatch ${JSON.stringify(s)}`);
 rows.push({...s,valid:!!valid,bitwiseEqual:true,scalarEqual:true,outputFnv64:fnv(base),outputSha256:sha(base)});
}
fs.writeFileSync(`${dir}/node-report.json`,JSON.stringify({wasmSha256:sha(wasm),node:process.version,rows},null,2)+'\n');console.log(`Verified ${rows.length} cases, including finite boundaries and Inf/NaN rejection.`);
