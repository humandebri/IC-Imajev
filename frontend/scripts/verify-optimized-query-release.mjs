/** Full-model, loopback-only comparison. Never edits release pins or deploys. */
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { HttpAgent, StatePaths } from '@icp-sdk/core/agent';
import { Principal } from '@icp-sdk/core/principal';
import { IDL } from '@icp-sdk/core/candid';
import { Tokenizer } from '@huggingface/tokenizers';
import { methods } from '../src/inference-agent.ts';
import { runQueryGraph } from '../src/query-runner.ts';
import { selectPlan, queryCount } from '../src/query-plan.ts';
import { tokenizeInput } from '../src/tokenization.ts';
import { validatePair } from './optimized-query-evidence.mjs';
import release from '../src/inference-release.json' with { type: 'json' };
const { values } = parseArgs({ options: Object.fromEntries(['reference-url', 'reference-canister', 'candidate-url', 'candidate-canister', 'candidate-module', 'directory', 'suffixes', 'concurrency'].map(name => [name, { type: 'string' }])) });
for (const name of ['reference-url', 'reference-canister', 'candidate-url', 'candidate-canister', 'candidate-module', 'directory']) assert(values[name], `--${name} required`);
assert(/^[0-9a-f]{64}$/.test(values['candidate-module']));
for (const name of ['reference-url', 'candidate-url']) {
 const u = new URL(values[name]); assert(u.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname), 'Only explicit loopback IC endpoints are accepted');
}
assert.notEqual(`${values['reference-url']}/${values['reference-canister']}`, `${values['candidate-url']}/${values['candidate-canister']}`);
const output = resolve(values.directory); await mkdir(output, { recursive: false });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const json = async name => JSON.parse(await readFile(new URL(name, import.meta.url)));
const manifestBytes = await readFile(new URL('../public/inference/prefix5-v1/manifest.json', import.meta.url));
assert.equal(sha(manifestBytes), release.manifest_sha256);
const manifest = JSON.parse(manifestBytes), assets = new Map();
for (let layer = 0; layer < 32; layer++) {
 const entry = manifest.assets[layer], b = await readFile(new URL(`../public/inference/prefix5-v1/${entry.file}`, import.meta.url));
 assert.equal(sha(b), entry.sha256); assets.set(layer, new Uint8Array(b));
}
const tokenizer = new Tokenizer(await json('../public/tokenizer/tokenizer.json'), await json('../public/tokenizer/tokenizer_config.json'));
const codes = (await json('../public/tokenizer/decision_readout.json')).codes.map(x => x.code);
const example = (await json('../src/boom-examples.generated.json'))[1];
const suffix = tokenizeInput(tokenizer, example.input, codes).tokenIds.slice(5);
async function client(label, moduleHash) {
 const agent = await HttpAgent.create({ host: values[`${label}-url`], retryTimes: 0, verifyQuerySignatures: true });
 await agent.fetchRootKey(); const canister = Principal.fromText(values[`${label}-canister`]);
 return {
  async checkModule() {
   const path = StatePaths.canisterModuleHash(canister), response = await agent.readState({ canisterId: canister }, { paths: [path] });
   assert.equal(Buffer.from(response.values.get(path) ?? []).toString('hex'), moduleHash, `${label} module mismatch`);
  },
  async query(method, args) {
   const schema = methods[method], request = IDL.encode(schema.args, args);
   const response = await agent.query(canister, { methodName: method, arg: request });
   assert.equal(response.status, 'replied'); const reply = response.reply.arg;
   assert(request.byteLength < 1_990_000 && reply.byteLength < 1_990_000, 'Wire budget exceeded');
   return { value: IDL.decode(schema.reply, reply)[0], requestBytes: request.byteLength, replyBytes: reply.byteLength };
  }
 };
}
const reference = await client('reference', release.module_hash), candidate = await client('candidate', values['candidate-module']);
await reference.checkModule(); await candidate.checkModule();
async function run(wire, n, moduleHash) {
 const rows = [], ids = [...manifest.prefix, ...Array.from({ length: n }, (_, i) => suffix[i % suffix.length])];
 const plan = selectPlan(n), signal = AbortSignal.timeout(600_000);
 let terminal;
 const client = { checkModule: wire.checkModule, async query(method, args) {
  const reply = await wire.query(method, args), value = reply.value;
  assert(value.Ok, `${method}: ${JSON.stringify(value.Err)}`);
  const measurement = value.Ok.measurement ?? value.Ok;
  rows.push({ method, instructions: Number(measurement.instructions), requestBytes: reply.requestBytes, replyBytes: reply.replyBytes });
  if (method === 'runFinalInferenceStep') terminal = value.Ok.decision;
  return value;
 } };
 await wire.checkModule();
 const result = await runQueryGraph(ids, example.input.options, {
  // The saved prefix inputs are held constant for this comparison only.
  // This does not create or approve a new release asset manifest.
  manifest: { ...manifest, module_hash: moduleHash }, asset: async layer => assets.get(layer), client, plan, signal, progress() {},
  observe(step, request, reply) {
   const header = new DataView(reply.buffer, reply.byteOffset, 4).getUint32(0, true);
   rows[step].requestHash = sha(request); rows[step].payloadHash = sha(reply.subarray(4 + header, -32));
  }
 });
 await wire.checkModule(); assert.equal(rows.length, queryCount(plan));
 const logits = Buffer.alloc(terminal.raw_logits.length * 4); terminal.raw_logits.forEach((x, i) => logits.writeFloatLE(x, i * 4));
 return { module: moduleHash, n, inputHash: sha(Buffer.from(JSON.stringify(ids))), plan, result, logitsBits: logits.toString('hex'), rows,
  totalInstructions: rows.reduce((sum, r) => sum + r.instructions, 0), maxInstructions: Math.max(...rows.map(r => r.instructions)) };
}
const wanted = values.suffixes ? values.suffixes.split(',').map(Number) : Array.from({ length: 91 }, (_, i) => i + 1);
assert(wanted.length === new Set(wanted).size && wanted.every(n => Number.isInteger(n) && n >= 1 && n <= 91));
const summary = { complete: false, query_gate_passed: false, release_eligible: false, reference_module: release.module_hash, candidate_module: values['candidate-module'], inputs: [], failures: [] };
const concurrency = Number(values.concurrency ?? 1);
assert(Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= 8);
let index = 0, stopped = false, writes = Promise.resolve();
function persist() {
 const snapshot = JSON.stringify(summary, null, 2) + '\n';
 writes = writes.then(() => writeFile(`${output}/report.json`, snapshot));
 return writes;
}
async function compare(n) {
 console.log(JSON.stringify({ n, phase: 'reference' })); const baseline = await run(reference, n, release.module_hash);
 console.log(JSON.stringify({ n, phase: 'candidate' })); const actual = await run(candidate, n, values['candidate-module']);
 const evidence = [baseline, actual].map(result => Buffer.from(JSON.stringify(result, null, 2)));
 await writeFile(`${output}/n${n}-reference.json`, evidence[0], { flag: 'wx' });
 await writeFile(`${output}/n${n}-candidate.json`, evidence[1], { flag: 'wx' });
 validatePair(evidence, n, release.module_hash, values['candidate-module']);
 assert.equal(actual.inputHash, baseline.inputHash); assert.equal(actual.logitsBits, baseline.logitsBits, 'Logits differ');
 assert.deepEqual(actual.rows.map(r => r.payloadHash), baseline.rows.map(r => r.payloadHash), 'Intermediate state bits differ');
 assert.deepEqual(actual.rows.map(r => r.requestHash), baseline.rows.map(r => r.requestHash), 'Input frames differ');
 assert.deepEqual(actual.result, baseline.result, 'Decision differs');
 assert(actual.maxInstructions <= 4_000_000_000, 'Query instruction budget exceeded');
 assert(actual.totalInstructions * 100 <= baseline.totalInstructions * 105, 'More than 5% instruction regression');
 summary.inputs.push({ n, reference: baseline.totalInstructions, candidate: actual.totalInstructions, evidence: evidence.map(sha) });
 console.log(JSON.stringify({n, phase: 'passed', ratio: actual.totalInstructions / baseline.totalInstructions}));
 await persist();
}
await Promise.all(Array.from({length: concurrency}, async () => {
 while (!stopped && index < wanted.length) {
  const n = wanted[index++];
  try { await compare(n); }
  catch (error) { stopped = true; summary.failures.push(`n=${n}: ${String(error)}`); process.exitCode = 1; await persist(); }
 }
}));
summary.inputs.sort((a,b) => a.n - b.n);
summary.complete = !stopped && wanted.length === 91 && summary.inputs.length === 91;
summary.query_gate_passed = summary.complete;
await persist();
console.log(JSON.stringify(summary));
