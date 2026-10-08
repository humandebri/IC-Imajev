/** Opt-in anonymous mainnet calibration. Never deploys, updates, or retries queries. */
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { Tokenizer } from "@huggingface/tokenizers";
import { tokenizeInput } from "../src/tokenization.ts";
import { createHash } from "node:crypto";
import { IDL } from "@icp-sdk/core/candid";
import { createQueryClient, methods } from "../src/inference-agent.ts";
import release from "../src/inference-release.json" with { type: "json" };
import { runQueryGraph } from "../src/query-runner.ts";
import { baselinePlan, queryCount, selectPlan } from "../src/query-plan.ts";
import { optimizeQueryPlan } from "./optimize-query-plan.mjs";
import { measurementOutput, readReusableMeasurement } from "./query-calibration-policy.mjs";
const root = new URL("../../", import.meta.url);
const calibration = new URL("artifacts/adaptive-query-20261008/", root);
const optimize = process.argv.includes("--optimize");
const balanced = process.argv.includes("--balanced");
const published = process.argv.includes("--published");
const real = process.argv.includes("--real");
const resume = process.argv.includes("--resume");
assert([optimize, balanced, published].filter(Boolean).length <= 1, "Choose one candidate strategy");
const output = measurementOutput(calibration, { resume, published });
console.log(JSON.stringify({measurementDirectory:output.pathname,resume}));
const record = JSON.parse(await readFile(new URL("artifacts/text-short-v2/inputs.json", root), "utf8")).records[2];
const manifestBytes=await readFile(new URL("frontend/public/inference/prefix27-v1/manifest.json", root));
assert.equal(createHash("sha256").update(manifestBytes).digest("hex"),release.manifest_sha256);
const manifest = JSON.parse(manifestBytes);
const assets = new Map();
for (let layer = 0; layer < 32; layer++) {
  const entry=manifest.assets[layer];
  const bytes=await readFile(new URL(`frontend/public/inference/prefix27-v1/${entry.file}`, root));
  assert.equal(bytes.length,entry.bytes);assert.equal(createHash("sha256").update(bytes).digest("hex"),entry.sha256);
  assets.set(layer,new Uint8Array(bytes));
}
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
export function candidatePlan(n) {
  const plan = baselinePlan(n);
  if (n >= 56) { plan.completions = plan.fronts.slice(0, 31).map(f => f + 1024); plan.fronts.fill(256); }
  return plan;
}
function referencePlan(n) {
  if (n <= 57) return baselinePlan(n);
  const plan = baselinePlan(n); plan.fronts.fill(256); plan.completions.fill(8960); return plan;
}
async function run(n, label, plan, ids, options) {
  const dir = new URL(`n${String(n).padStart(2, "0")}-${label}/`, output);
  await mkdir(dir, { recursive: true });
  const file = new URL("report.json", dir);
  const old=await readReusableMeasurement(file,{moduleHash:release.module_hash,plan,inputHash:hash(Buffer.from(JSON.stringify(ids))),options},{resume,published});
  if(old) {
    console.log(JSON.stringify({n,label,reusedCalibration:file.pathname,note:"Cached timings are not a new performance measurement"}));
    return old;
  }
  const signal = AbortSignal.timeout(600_000);
  const client = await createQueryClient(signal);
  await client.checkModule();
  const rows = []; let terminal; let finalState;
  const started = performance.now();
  const measured = { ...client, async query(method, args) {
    const requestBytes = IDL.encode(methods[method].args, args).byteLength;
    const clock = performance.now(); const value = await client.query(method, args);
    const measurement = value.Ok?.measurement ?? value.Ok;
    rows.push({ method, instructions: Number(measurement?.instructions ?? 0),
      requestBytes, replyBytes: IDL.encode(methods[method].reply, [value]).byteLength,
      seconds: (performance.now() - clock) / 1000,
      spans: measurement?.spans?.map(([label, n]) => [label, Number(n)]), heapPages: Number(measurement?.heap_pages ?? 0) });
    if (rows.length % 16 === 0) console.log(JSON.stringify({ n, label, completed: rows.length, total: queryCount(plan) }));
    await writeFile(new URL("progress.json", dir), JSON.stringify({ n, label, completed: rows.length, total: queryCount(plan) }));
    if (method === "terminal_step_decision") {
      terminal = value.Ok?.decision;
      await writeFile(new URL("terminal-request.bin", dir), args[0]);
    }
    return value;
  } };
  try {
    const result = await runQueryGraph(ids, options, { manifest, asset: async layer => assets.get(layer), client: measured, plan, signal,
      progress() {}, observe(step, request, reply) {
        const size = new DataView(reply.buffer, reply.byteOffset, 4).getUint32(0, true);
        rows[step].requestHash = hash(request); rows[step].payloadHash = hash(reply.subarray(4 + size, -32));
        if (step === queryCount(plan) - 1) finalState = reply.slice();
      } });
    await client.checkModule();
    await writeFile(new URL("terminal.bin", dir), finalState);
    const report = { complete: true, moduleHash: release.module_hash, n, label, plan, options,
      inputHash: hash(Buffer.from(JSON.stringify(ids))), result, rawLogits: terminal.raw_logits,
      queries: rows, count: rows.length, seconds: (performance.now() - started) / 1000,
      maxInstructions: Math.max(...rows.map(q => q.instructions)), totalInstructions: rows.reduce((sum,q)=>sum+q.instructions,0),
      maxRequestBytes: Math.max(...rows.map(q=>q.requestBytes)), maxReplyBytes: Math.max(...rows.map(q=>q.replyBytes)),
      totalBytes: rows.reduce((sum,q)=>sum+q.requestBytes+q.replyBytes,0),
      finalPayloadHash: rows.at(-1).payloadHash };
    await writeFile(file, JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify({ n, label, count: report.count, maxInstructions: report.maxInstructions, seconds: report.seconds }));
    return report;
  } catch(error) {
    await writeFile(new URL("failure.json",dir),JSON.stringify({n,label,plan,queries:rows,error:String(error)},null,2));
    throw error;
  }
}
await mkdir(output, { recursive: true });
async function optimizedPlan(n) {
  assert(n>=56 && n<=69, "Optimization requires paired measurements");
  const stem=`n${String(n).padStart(2,"0")}`;
  const baseline=JSON.parse(await readFile(new URL(`${stem}-reference/report.json`, calibration),"utf8"));
  const split=JSON.parse(await readFile(new URL(`${stem}-candidate/report.json`, calibration),"utf8"));
  assert.equal(split.count,63);
  const probeBase=JSON.parse(await readFile(new URL("n57-reference/report.json", calibration),"utf8"));
  const probeSplit=JSON.parse(await readFile(new URL("n57-candidate/report.json", calibration),"utf8"));
  if (balanced) {
    const plan=optimizeQueryPlan(baseline,split,probeBase,probeSplit,baselinePlan);
    console.log(JSON.stringify({n,label:"balanced-plan",count:queryCount(plan),plan}));
    return plan;
  }
  const plan=baselinePlan(n), target=3_850_000_000;
  const unit0=(probeBase.queries[0].instructions-probeSplit.queries[0].instructions)/(probeBase.plan.fronts[0]-probeSplit.plan.fronts[0])*n/57;
  plan.fronts[0]=Math.min(8960,Math.floor((baseline.plan.fronts[0]+(target-baseline.queries[0].instructions)/unit0)/256)*256);
  for(let layer=0;layer<=30;layer++) {
    const a=baseline.queries[baseline.count===32?layer+1:2*layer+2], b=split.queries[2*layer+2];
    const terminal=layer===30;
    const originalNext=terminal?0:baseline.plan.fronts[layer+1];
    const splitNext=terminal?0:split.plan.fronts[layer+1];
    const unit=(a.instructions-b.instructions)/(split.plan.completions[layer]-baseline.plan.completions[layer]+originalNext-splitNext);
    assert(unit>0 && Number.isFinite(unit));
    let done=plan.fronts[layer];
    const limit = () => originalNext+done-baseline.plan.completions[layer]+(target-a.instructions)/unit;
    if(!terminal && limit()<256) done=8960;
    plan.completions[layer]=done;
    if(!terminal)plan.fronts[layer+1]=Math.min(8960,Math.floor(limit()/256)*256);
    else assert(a.instructions+unit*(baseline.plan.completions[layer]-done)<target);
  }
  console.log(JSON.stringify({n,label:"optimized-plan",count:queryCount(plan),plan}));
  return plan;
}
const wanted = process.argv.slice(2).filter(s=>!s.startsWith("--")).map(Number);
const ns = wanted.length ? wanted : Array.from({length:69},(_,i)=>i+1);
for (const n of ns) {
  assert(Number.isInteger(n) && n >= 1 && n <= 69);
  // Exact token-length protocol fixtures; do not claim these are all user prompts.
  const suffix = record.token_ids.slice(27);
  let ids = [...manifest.prefix, ...Array.from({length:n},(_,i)=>suffix[i % suffix.length])];
  let options = record.options;
  if(real) {
    assert(n>=57);
    const tokenizer=new Tokenizer(JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer.json", root))),JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer_config.json", root))));
    const readout=JSON.parse(await readFile(new URL("frontend/public/tokenizer/decision_readout.json", root)));
    const input={state:"Minimum voting dissolve delay changes from 1 day to 2 days."+" more".repeat(n-42),question:"Approve?",options:["no","yes"]};
    options=input.options;
    await writeFile(new URL(`real-input-n${n}.json`,output),JSON.stringify(input,null,2));
    ids=tokenizeInput(tokenizer,input,readout.codes.map(e=>e.code)).tokenIds;
    assert.equal(ids.length,n+27);
  }
  const label = `${real ? "real-" : ""}${published ? "published" : balanced ? "balanced" : optimize ? "optimized" : "candidate"}`;
  const candidate = await run(n, label, published ? selectPlan(n) : optimize || balanced ? await optimizedPlan(n) : candidatePlan(n), ids, options);
  const reference = JSON.stringify(candidate.plan) === JSON.stringify(referencePlan(n)) ? candidate : await run(n, real ? "real-reference" : "reference", referencePlan(n), ids, options);
  assert.equal(candidate.finalPayloadHash, reference.finalPayloadHash, `final hidden/KV bits n=${n}`);
  assert.deepEqual(candidate.result, reference.result, `decision bits n=${n}`);
  assert.deepEqual(candidate.rawLogits, reference.rawLogits, `logit bits n=${n}`);
  assert(candidate.maxInstructions <= 4_000_000_000, `handler budget n=${n}: ${candidate.maxInstructions}`);
  assert(candidate.maxRequestBytes < 1_990_000 && candidate.maxReplyBytes < 1_990_000);
  assert(candidate.queries.every(q=>q.heapPages*65536<2**32), `heap budget n=${n}`);
  await writeFile(new URL(`${real?"real-":""}${published?"published-":balanced?"balanced-":""}n${String(n).padStart(2,"0")}-verified.json`, output), JSON.stringify({ n, candidate, reference, referenceKind: candidate === reference ? "unchanged-baseline" : "separate-query-schedule", bitwiseEqual:true }, null,2)+"\n");
}
