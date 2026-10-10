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
import { optimizeStreamQueryPlan } from "./optimize-query-plan.mjs";
import { measurementOutput, readReusableMeasurement } from "./query-calibration-policy.mjs";
const root = new URL("../../", import.meta.url);
const calibration = new URL("artifacts/prefix5-calibration/", root);
const optimize = process.argv.includes("--optimize");
const balanced = process.argv.includes("--balanced");
const published = process.argv.includes("--published");
const real = process.argv.includes("--real");
const resume = process.argv.includes("--resume");
assert([optimize, balanced, published].filter(Boolean).length <= 1, "Choose one candidate strategy");
const output = measurementOutput(calibration, { resume, published });
console.log(JSON.stringify({measurementDirectory:output.pathname,resume}));
const example = JSON.parse(await readFile(new URL("frontend/src/boom-examples.generated.json", root), "utf8"))[1];
const fixtureTokenizer = new Tokenizer(JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer.json", root))), JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer_config.json", root))));
const fixtureReadout = JSON.parse(await readFile(new URL("frontend/public/tokenizer/decision_readout.json", root)));
const record = {token_ids: tokenizeInput(fixtureTokenizer, example.input, fixtureReadout.codes.map(e => e.code)).tokenIds, options: example.input.options};
const manifestBytes=await readFile(new URL("frontend/public/inference/prefix5-v1/manifest.json", root));
assert.equal(createHash("sha256").update(manifestBytes).digest("hex"),release.manifest_sha256);
const manifest = JSON.parse(manifestBytes);
const assets = new Map();
for (let layer = 0; layer < 32; layer++) {
  const entry=manifest.assets[layer];
  const bytes=await readFile(new URL(`frontend/public/inference/prefix5-v1/${entry.file}`, root));
  assert.equal(bytes.length,entry.bytes);assert.equal(createHash("sha256").update(bytes).digest("hex"),entry.sha256);
  assets.set(layer,new Uint8Array(bytes));
}
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
export function candidatePlan(n) { return baselinePlan(n); }
function referencePlan(n) { return baselinePlan(n); }
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
    const headerSize = new DataView(args[0].buffer, args[0].byteOffset, 4).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(args[0].subarray(4, 4 + headerSize)));
    rows.push({ method, op: header.op, dims: header.dims, layer: Number(header.tensor.split(".")[3]), instructions: Number(measurement?.instructions ?? 0),
      requestBytes, replyBytes: IDL.encode(methods[method].reply, [value]).byteLength,
      seconds: (performance.now() - clock) / 1000,
      spans: measurement?.spans?.map(([label, n]) => [label, Number(n)]), heapPages: Number(measurement?.heap_pages ?? 0) });
    if (rows.length % 16 === 0) console.log(JSON.stringify({ n, label, completed: rows.length, total: queryCount(plan) }));
    await writeFile(new URL("progress.json", dir), JSON.stringify({ n, label, completed: rows.length, total: queryCount(plan) }));
    if (method === "runFinalInferenceStep") {
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
const wanted = process.argv.slice(2).filter(s=>!s.startsWith("--")).map(Number);
const ns = wanted.length ? wanted : Array.from({length:91},(_,i)=>i+1);
for (const n of ns) {
  assert(Number.isInteger(n) && n >= 1 && n <= 91);
  // Exact token-length protocol fixtures; do not claim these are all user prompts.
  const suffix = record.token_ids.slice(manifest.prefix.length);
  let ids = [...manifest.prefix, ...Array.from({length:n},(_,i)=>suffix[i % suffix.length])];
  let options = record.options;
  if(real) {
    assert(n>=57);
    const tokenizer=new Tokenizer(JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer.json", root))),JSON.parse(await readFile(new URL("frontend/public/tokenizer/tokenizer_config.json", root))));
    const readout=JSON.parse(await readFile(new URL("frontend/public/tokenizer/decision_readout.json", root)));
    const input={state:"Minimum voting dissolve delay changes from 1 day to 2 days."+" more".repeat(Math.max(0,n-43)),question:"Approve?",options:["no","yes"]};
    options=input.options;
    await writeFile(new URL(`real-input-n${n}.json`,output),JSON.stringify(input,null,2));
    ids=tokenizeInput(tokenizer,input,readout.codes.map(e=>e.code)).tokenIds;
    assert.equal(ids.length,n+manifest.prefix.length);
  }
  const label = `${real ? "real-" : ""}${published ? "published" : balanced ? "balanced" : optimize ? "optimized" : "candidate"}`;
  const reference = await run(n, real ? "real-reference" : "reference", referencePlan(n), ids, options);
  const plan = published ? selectPlan(n) : (optimize || balanced) && n > 45
    ? optimizeStreamQueryPlan(reference, baselinePlan) : candidatePlan(n);
  const candidate = JSON.stringify(plan) === JSON.stringify(reference.plan) ? reference
    : await run(n, label, plan, ids, options);
  assert.equal(candidate.finalPayloadHash, reference.finalPayloadHash, `final hidden/KV bits n=${n}`);
  assert.deepEqual(candidate.result, reference.result, `decision bits n=${n}`);
  assert.deepEqual(candidate.rawLogits, reference.rawLogits, `logit bits n=${n}`);
  assert(candidate.maxInstructions <= 4_000_000_000, `handler budget n=${n}: ${candidate.maxInstructions}`);
  assert(candidate.maxRequestBytes < 1_990_000 && candidate.maxReplyBytes < 1_990_000);
  assert(candidate.queries.every(q=>q.heapPages*65536<2**32), `heap budget n=${n}`);
  await writeFile(new URL(`${real?"real-":""}${published?"published-":balanced?"balanced-":""}n${String(n).padStart(2,"0")}-verified.json`, output), JSON.stringify({ n, candidate, reference, referenceKind: candidate === reference ? "unchanged-baseline" : "separate-query-schedule", bitwiseEqual:true }, null,2)+"\n");
}
