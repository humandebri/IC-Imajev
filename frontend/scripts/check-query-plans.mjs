import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runQueryGraph } from "../src/query-runner.ts";
import { baselinePlan, queryCount, selectPlan, validatePlan, MAX_SUFFIX } from "../src/query-plan.ts";
import { frame, concat, u32, C, CONV } from "../src/query-codec.ts";
const profiles=JSON.parse(await readFile(new URL("../src/query-profiles.json",import.meta.url),"utf8"));
const manifest = JSON.parse(await readFile(new URL("../public/inference/prefix27-v1/manifest.json",import.meta.url),"utf8"));
const carry = (n, done) => concat(new Uint8Array([1]),u32(done),new Uint8Array(n*(3*C+done+4*(C/256+done/256+192))));
function fixture(n, plan, failure, signal = new AbortController().signal) {
  const seen = []; const progress = [];
  const ids = [...manifest.prefix, ...Array(n).fill(198)];
  const d = { manifest, plan, signal, asset: async () => new Uint8Array(0), progress: (completed,total) => progress.push([completed,total]),
    client: { async checkModule() {}, async query(method,args) {
      const request=args[0];const size=new DataView(request.buffer,request.byteOffset,4).getUint32(0,true);
      const h=JSON.parse(new TextDecoder().decode(request.subarray(4,4+size))); const layer=Number(h.tensor.split(".")[3]);
      assert.equal(h.step,seen.length);assert.equal(h.input_hash.length,64);seen.push({method,layer,op:h.op});
      if (failure === "timeout" && seen.length===2) throw new DOMException("Timed out","TimeoutError");
      const out={...h,step:h.step+1};let payload;
      if(method==="terminal_step_decision") return {Ok:{measurement:{state:await frame(out,concat(new Uint8Array([0]),new Uint8Array(2*(2*C+n*2048))))},decision:{value:["yes"],probabilities:[0.6,0.3],unknown_probability:0.1,abstained:false}}};
      if(method==="step") {
        const done=h.op==="mlp_stream_next" ? h.dims[1]+h.dims[2] : h.dims[1];payload=carry(n,done);
        if(h.op==="delta_mlp_stream_start_ids")payload=concat(new Uint8Array([0]),payload,new Uint8Array(2*CONV));
      } else {
        out.tensor=`model.language_model.layers.${layer+1}.post_attention_layernorm.weight`;
        out.aux=[`model.language_model.layers.${layer+2}.input_layernorm.weight`];out.op="mlp_stream_prepare";out.encoding="mlp-stream-exact-v1";out.dims=[n,0,plan.fronts[layer+1]];
        payload=carry(n,plan.fronts[layer+1]);
      }
      if (failure === "carry" && h.op==="mlp_stream_next") payload[1]^=1;
      if (failure === "step" && h.op==="mlp_stream_next") out.step++;
      return {Ok:{state:await frame(out,payload),previous_hidden:new Uint8Array(2*n*C),conv:new Uint8Array(2*CONV),kv:new Uint8Array(2*n*2048)}};
    } } };
  return {ids,d,seen,progress};
}
const makePlan = n => {const p=baselinePlan(n);p.completions=p.fronts.slice(0,31).map((f,layer)=>f+(layer%2===0?256:0));return p;};
for(const n of [1,57,69]) {
  const plan=makePlan(n),t=fixture(n,plan);await runQueryGraph(t.ids,["yes","no"],t.d);
  assert.equal(t.seen.length,48);assert.equal(queryCount(plan),48);
  assert.deepEqual(t.progress,Array.from({length:49},(_,i)=>[i,48]));
  assert.equal(t.seen.at(-2).op,"mlp_stream_next");assert.equal(t.seen.at(-1).method,"terminal_step_decision");
}
const concurrent=await Promise.all([1,69].map(async n=>{const t=fixture(n,makePlan(n));await runQueryGraph(t.ids,["yes","no"],t.d);return t;}));
assert.equal(concurrent[0].seen.length,48);assert.equal(concurrent[1].seen.length,48);
for(const [failure,message] of [["carry",/carry/],["step",/step mismatch/],["timeout",/Timed out/]]) {
  const t=fixture(1,makePlan(1),failure);await assert.rejects(runQueryGraph(t.ids,["yes","no"],t.d),message);assert.equal(t.seen.length,2);
}
const cancelled=new AbortController(),t=fixture(1,makePlan(1),null,cancelled.signal);
t.d.progress=(n)=>{if(n===2)cancelled.abort();};await assert.rejects(runQueryGraph(t.ids,["yes","no"],t.d),/abort/i);assert.equal(t.seen.length,2);
assert.throws(()=>selectPlan(1,"wrong"),/runtime/);
assert.throws(()=>selectPlan(MAX_SUFFIX+1),/exceeds/);
for(const mutate of [p=>p.fronts[0]=257,p=>p.completions[0]=9216,p=>p.moduleHash="wrong",p=>p.suffix++,p=>p.fronts.pop(),p=>p.completions[0]=256]) {
  const p=makePlan(1);mutate(p);assert.throws(()=>validatePlan(p,1),/Invalid/);
}
for(let n=1;n<=MAX_SUFFIX;n++) {
  const plan=selectPlan(n);
  assert.equal(queryCount(plan),profiles.evidence[String(n)].queries);
  assert(profiles.evidence[String(n)].maxInstructions<=4_000_000_000);
}
console.log("Variable step/layer binding, terminal numbering, dynamic totals, concurrent carries, cancellation, timeout, corrupt carry, invalid plans and runtime pin passed.");
