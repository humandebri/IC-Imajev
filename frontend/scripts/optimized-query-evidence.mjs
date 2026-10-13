import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { queryCount } from '../src/query-plan.ts';

export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export function validatePair(raw, n, reference, candidate, recorded) {
 const [baseline, actual] = raw.map(bytes => JSON.parse(bytes));
 if (recorded) {
  assert.equal(recorded.n, n);
  assert.deepEqual(raw.map(sha), recorded.evidence, `Evidence bytes changed at suffix ${n}`);
 }
 assert.equal(baseline.module, reference); assert.equal(actual.module, candidate);
 assert.equal(baseline.n, n); assert.equal(actual.n, n);
 assert.equal(actual.inputHash, baseline.inputHash); assert.equal(actual.logitsBits, baseline.logitsBits);
 assert.deepEqual(actual.result, baseline.result); assert.deepEqual(actual.plan, baseline.plan);
 for (const field of ['payloadHash', 'requestHash', 'method']) assert.deepEqual(actual.rows.map(r => r[field]), baseline.rows.map(r => r[field]));
 for (const result of [baseline, actual]) {
  assert.equal(result.rows.length, queryCount(result.plan));
  assert(/^[0-9a-f]{64}$/.test(result.inputHash));
  for (const row of result.rows) {
   assert(Number.isSafeInteger(row.instructions) && row.instructions > 0 && row.instructions <= 4_000_000_000);
   for (const bytes of [row.requestBytes, row.replyBytes]) assert(Number.isSafeInteger(bytes) && bytes > 0 && bytes < 1_990_000);
   assert(/^[0-9a-f]{64}$/.test(row.payloadHash) && /^[0-9a-f]{64}$/.test(row.requestHash));
  }
  assert.equal(result.totalInstructions, result.rows.reduce((sum, row) => sum + row.instructions, 0));
  assert.equal(result.maxInstructions, Math.max(...result.rows.map(row => row.instructions)));
 }
 assert(actual.totalInstructions * 100 <= baseline.totalInstructions * 105);
 if (recorded) {
  assert.equal(recorded.reference, baseline.totalInstructions);
  assert.equal(recorded.candidate, actual.totalInstructions);
 }
 return { baseline, actual };
}

export async function loadValidatedEvidence(directory, summary, module) {
 assert(summary.complete && summary.query_gate_passed && !summary.failures?.length);
 assert.equal(summary.candidate_module, module);
 assert.equal(summary.inputs.length, 91);
 assert.deepEqual(summary.inputs.map(row => row.n).sort((a,b) => a-b), Array.from({length:91}, (_, i) => i+1));
 const validated = [];
 for (let n=1; n<=91; n++) {
  const raw = await Promise.all(['reference', 'candidate'].map(label => readFile(`${directory}/n${n}-${label}.json`)));
  const pair = validatePair(raw, n, summary.reference_module, module, summary.inputs.find(row => row.n === n));
  validated.push({ ...pair, bytes: raw[1] });
 }
 return validated;
}
