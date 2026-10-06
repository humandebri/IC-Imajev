import fs from 'node:fs';
import crypto from 'node:crypto';
import {performance} from 'node:perf_hooks';
const [dir] = process.argv.slice(2);
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const fnv = b => {let h=14695981039346656037n;for(const x of b)h=BigInt.asUintN(64,(h^BigInt(x))*1099511628211n);return h.toString(16).padStart(16,'0');};
const bytes=fs.readFileSync(`${dir}/kernels.wasm`);
const imports={ic0:{performance_counter:()=>0n,msg_arg_data_size:()=>0,msg_arg_data_copy:()=>{},msg_reply_data_append:()=>{},msg_reply:()=>{}}};
const {instance}=await WebAssembly.instantiate(bytes,imports);
const e=instance.exports;
const output=()=>Buffer.from(new Uint8Array(e.memory.buffer,e.output_ptr(),e.output_len()*4));
const median=v=>[...v].sort((a,b)=>a-b)[Math.floor(v.length/2)];
const cases=[];
for(const cols of [1024,2624]) {
  const data=fs.readFileSync(`${dir}/${cols===1024?'qkv':'wo'}.bin`);
  const weights=new Int8Array(data.buffer,data.byteOffset,512*cols);
  const sw=new Float32Array(data.buffer,data.byteOffset+512*cols,512);
  for(const tokens of [1,7,32,65,128]) {
    e.setup(tokens,512,cols);
    e.run(0);const baseline=output();e.run(1);
    if(!baseline.equals(output()))throw new Error(`backend mismatch ${tokens}/${cols}`);
    const ref=Buffer.alloc(tokens*512*4);
    for(let t=0;t<tokens;t++)for(let r=0;r<512;r++) {
      let sum=0;
      for(let c=0;c<cols;c++)sum+=((t*cols+c)*19%255-127)*weights[r*cols+c];
      const sx=Math.fround((t+1)/257);
      ref.writeFloatLE(Math.fround(Math.fround(Math.fround(sum)*sx)*sw[r])+0,(t*512+r)*4);
    }
    if(!baseline.equals(ref))throw new Error(`scalar mismatch ${tokens}/${cols}`);
    for(let i=0;i<8;i++){e.run(0);e.run(1);}
    const samples={laya:[],imajev_token:[]};
    for(let i=0;i<7;i++)for(const backend of (i%2?[1,0]:[0,1])) {
      const start=performance.now();e.run(backend);
      samples[backend===0?'laya':'imajev_token'].push(performance.now()-start);
    }
    e.run(0);if(!baseline.equals(output()))throw new Error('baseline changed');
    e.run(1);if(!baseline.equals(output()))throw new Error('candidate changed');
    const row={tokens,rows:512,cols,paddedCols:Math.ceil(cols/256)*256,bitwiseEqual:true,scalarEqual:true,
      outputSha256:sha(baseline),outputFnv64Hex:fnv(baseline),samplesMs:samples,
      medianMs:Object.fromEntries(Object.entries(samples).map(([name,v])=>[name,median(v)]))};
    cases.push(row);console.log(JSON.stringify({tokens,cols,medianMs:row.medianMs}));
  }
}
fs.writeFileSync(`${dir}/node-report.json`,JSON.stringify({scope:'Prepared dot + F32 writeback, real weight slices, synthetic activations; not full inference',node:process.version,nodeFlags:process.execArgv,wasmSha256:sha(bytes),rows:cases},null,2)+'\n');
