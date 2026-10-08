/** Read-only option-count and malformed-carry checks on measured terminal inputs. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createQueryClient } from "../src/inference-agent.ts";
import { frame, unframe } from "../src/query-codec.ts";
const root=new URL("../../artifacts/adaptive-query-20261008/",import.meta.url);
const requests=await Promise.all(["n69-real-optimized/terminal-request.bin","n69-real-reference/terminal-request.bin"].map(p=>readFile(new URL(p,root))));
const signal=AbortSignal.timeout(120_000),client=await createQueryClient(signal);await client.checkModule();
const rows=[];
for(let count=2;count<=7;count++) {
  const options=Array.from({length:count},(_,i)=>`候補${i+1}`);
  const replies=await Promise.all(requests.map(async request=>{
    const value=await client.query("terminal_step_decision",[new Uint8Array(request),options]);assert(value.Ok,value.Err);
    const m=value.Ok.measurement;assert(Number(m.instructions)<4_000_000_000);
    const size=request.readUInt32LE(0),header=JSON.parse(request.subarray(4,4+size));
    const payload=await unframe(m.state,{...header,step:header.step+1},true);
    return {decision:value.Ok.decision,payload:Buffer.from(payload),instructions:Number(m.instructions)};
  }));
  assert.deepEqual(replies[0].decision.probabilities,replies[1].decision.probabilities);
  assert.deepEqual(replies[0].decision.raw_logits,replies[1].decision.raw_logits);
  assert.equal(replies[0].decision.unknown_probability,replies[1].decision.unknown_probability);
  assert.deepEqual(replies[0].decision.value,replies[1].decision.value);
  assert.deepEqual(replies[0].payload,replies[1].payload);
  rows.push({count,bitwiseEqual:true,instructions:replies.map(r=>r.instructions)});
}
const bad=requests[0];const size=bad.readUInt32LE(0),header=JSON.parse(bad.subarray(4,4+size));
const payload=new Uint8Array(bad.subarray(4+size,-32));payload[6]^=1;
const rejected=await client.query("terminal_step_decision",[await frame(header,payload),["yes","no"]]);
assert(rejected.Err,"A signed but malformed carry must be rejected by the canister");
await client.checkModule();
await writeFile(new URL("terminal-options-report.json",root),JSON.stringify({rows,malformedCarryRejected:rejected.Err},null,2));
console.log("Option counts2–7 with Japanese UTF-8 labels are bit-identical across schedules; malformed terminal carry rejected.");
