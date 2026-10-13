// Revalidate complete per-input evidence after interrupted loopback measurement runs.
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { validatePair } from './optimized-query-evidence.mjs';

const { values, positionals } = parseArgs({ options: { directory: { type: 'string' }, module: { type: 'string' } }, allowPositionals: true });
assert(values.directory && /^[0-9a-f]{64}$/.test(values.module) && positionals.length);
const reference = '8fd78b39b4ab626b4623826c162275737d875b7df25341d1aa2852d6e50a546c';
const output = resolve(values.directory); await mkdir(output, { recursive: false });
const inputs = [];
for (let n = 1; n <= 91; n++) {
 let pair;
 for (const directory of positionals) {
  const summary = JSON.parse(await readFile(`${directory}/report.json`));
  assert.equal(summary.candidate_module, values.module); assert.equal(summary.reference_module, reference);
  if (!summary.inputs.some(row => row.n === n)) continue;
  const raw = await Promise.all(['reference', 'candidate'].map(label => readFile(`${directory}/n${n}-${label}.json`)));
  const { baseline, actual } = validatePair(raw, n, reference, values.module);
  const count = x => x.totalInstructions;
  pair = { n, reference: count(baseline), candidate: count(actual), evidence: raw.map(bytes => createHash('sha256').update(bytes).digest('hex')) };
  for (let i = 0; i < 2; i++) await writeFile(`${output}/n${n}-${i ? 'candidate' : 'reference'}.json`, raw[i]);
  break;
 }
 assert(pair, `Missing passed evidence for suffix ${n}`); inputs.push(pair);
}
await writeFile(`${output}/report.json`, JSON.stringify({ complete: true, query_gate_passed: true, release_eligible: false, reference_module: reference, candidate_module: values.module, inputs, origin: 'Revalidated original measurement bytes; interrupted attempts retained separately' }, null, 2) + '\n');
