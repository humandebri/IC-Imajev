/** Publish only a continuous, measured range pinned to this exact module. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import release from "../src/inference-release.json" with { type: "json" };
import { baselinePlan, queryCount, validatePlan } from "../src/query-plan.ts";
const root = new URL("../../", import.meta.url);
const plans = {}, evidence = {};
const balanced = process.argv.includes("--balanced");
const throughArg=process.argv.find(arg=>arg.startsWith("--through="));
const through=throughArg?Number(throughArg.slice("--through=".length)):69;
assert(Number.isInteger(through) && through>=58 && through<=69 && (!throughArg || balanced),"--through requires --balanced and a suffix length58–69");
for (let n=1;n<=69;n++) {
  const stem=`n${String(n).padStart(2,"0")}`;
  const optimized=balanced && n>=58 && n<=through;
  const reportURL=new URL(`artifacts/adaptive-query-20261008/${optimized ? "balanced-" : ""}${stem}-verified.json`,root);
  let bytes;
  try {bytes=await readFile(reportURL);} catch(e) {if(e.code==="ENOENT" && !balanced)break;throw e;}
  const {candidate:c,reference:r,bitwiseEqual}=JSON.parse(bytes);
  assert(c.complete && r.complete && bitwiseEqual && c.moduleHash===release.module_hash && r.moduleHash===release.module_hash);
  validatePlan(c.plan,n);assert.equal(c.count,queryCount(c.plan));
  assert(c.maxInstructions<=4_000_000_000 && c.maxRequestBytes<1_990_000 && c.maxReplyBytes<1_990_000);
  assert(c.queries.every(q=>q.heapPages*65536<2**32));
  assert.equal(c.finalPayloadHash,r.finalPayloadHash);assert.deepEqual(c.result,r.result);assert.deepEqual(c.rawLogits,r.rawLogits);
  const unchanged=JSON.stringify(c.plan)===JSON.stringify(baselinePlan(n));
  if(!unchanged)assert.notEqual(c.label,r.label,"Changed schedules require independent reference execution");
  if(optimized) {
    const previous=JSON.parse(await readFile(new URL(`artifacts/adaptive-query-20261008/${stem}-candidate/report.json`,root),"utf8"));
    assert.equal(c.label,"balanced");
    assert.equal(previous.moduleHash,c.moduleHash);assert.equal(previous.inputHash,c.inputHash);
    assert(c.count<previous.count && c.totalBytes<previous.totalBytes,"Optimization must reduce both queries and measured Candid bytes");
  }
  plans[n]={fronts:c.plan.fronts,completions:c.plan.completions};
  evidence[n]={queries:c.count,maxInstructions:c.maxInstructions,maxRequestBytes:c.maxRequestBytes,maxReplyBytes:c.maxReplyBytes,
    reference:unchanged?"unchanged-baseline":"separate-query-schedule",finalPayloadHash:c.finalPayloadHash,
    reportSHA256:createHash("sha256").update(bytes).digest("hex")};
}
const maxSuffix=Object.keys(plans).length;
assert(maxSuffix>=57,"Never regress the existing input range because calibration is incomplete");
const value={moduleHash:release.module_hash,maxSuffix,targetInstructions:4_000_000_000,plans,evidence};
const dest=new URL("../src/query-profiles.json",import.meta.url);
const encoded=JSON.stringify(value,null,2)+"\n";
if(process.argv.includes("--check"))assert(await readFile(dest,"utf8")===encoded,"Query profile evidence changed");
else await writeFile(dest,encoded);
console.log(`Verified execution profiles: suffix1–${maxSuffix}, total${maxSuffix+27}; module${release.module_hash}.`);
