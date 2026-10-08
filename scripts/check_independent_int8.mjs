// Compare both real Wasm revisions under the same fixture, flags and host engine.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const [beforePath,afterPath,reportPath]=process.argv.slice(2);
assert(beforePath && afterPath && reportPath,'Pass baseline.wasm candidate.wasm report.json');
const imports={ic0:{performance_counter:()=>0n,msg_arg_data_size:()=>0,msg_arg_data_copy:()=>{},msg_reply_data_append:()=>{},msg_reply:()=>{}}};
const instances=await Promise.all([beforePath,afterPath].map(async path=>(await WebAssembly.instantiate(await readFile(path),imports)).instance));
const output=i=>{const e=i.exports;return Buffer.from(new Uint8Array(e.memory.buffer,e.output_ptr(),e.output_len()*4));};
const fnv=b=>{let value=14695981039346656037n;for(const byte of b)value=BigInt.asUintN(64,(value^BigInt(byte))*1099511628211n);return value.toString(16).padStart(16,'0');};
const hash=b=>createHash('sha256').update(b).digest('hex');
const median=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
const rows=[];
for(const mode of [0]) {
  for(const n of Array.from({length:132},(_,i)=>i+1)) {
    for(const cols of [256,512,2560,9216]) for(let pattern=0;pattern<6;pattern++) {
      for(const i of instances){i.exports.setup(mode,n,cols,pattern);i.exports.run();}
      const reference=output(instances[0]);assert(reference.equals(output(instances[1])),`bit mismatch mode${mode} n${n} cols${cols} pattern${pattern}`);
      const row={mode,tokens:n,cols,pattern,bitwiseEqual:true,outputSha256:hash(reference),outputFnv64:fnv(reference)};
      if(pattern===0 && [1,69,87,132].includes(n) && cols!==512) {
        for(let warm=0;warm<3;warm++)for(const i of instances)i.exports.run();
        const samples=[[],[]];
        for(let sample=0;sample<5;sample++) for(const k of sample%2?[1,0]:[0,1]) {
          const start=performance.now();instances[k].exports.run();samples[k].push(performance.now()-start);
        }
        row.samplesMs=samples;row.medianMs=samples.map(median);row.afterOverBefore=row.medianMs[1]/row.medianMs[0];
        assert(reference.equals(output(instances[0])) && reference.equals(output(instances[1])));
      }
      rows.push(row);
    }
  }
  console.log(JSON.stringify({mode,bitwiseCases:rows.length}));
}
const report={scope:'Synthetic projection A/B in Node/V8; not IC instruction counts or full-model proof',node:process.version,nodeFlags:process.execArgv,baselineSHA256:hash(await readFile(beforePath)),candidateSHA256:hash(await readFile(afterPath)),complete:true,rows};
await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({complete:true,bitwiseCases:rows.length,maxMedianRatio:Math.max(...rows.filter(r=>r.afterOverBefore).map(r=>r.afterOverBefore))}));
