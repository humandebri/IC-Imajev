import assert from 'node:assert/strict';
import { validatePair, sha, loadValidatedEvidence } from './optimized-query-evidence.mjs';
import { selectPlan, queryCount } from '../src/query-plan.ts';
import release from '../src/inference-release.json' with { type: 'json' };

const module = 'b'.repeat(64), plan = selectPlan(1);
const rows = Array.from({length: queryCount(plan)}, () => ({method:'query', instructions:100, requestBytes:10, replyBytes:10, requestHash:'a'.repeat(64), payloadHash:'c'.repeat(64)}));
const baseline = {module:release.module_hash, n:1, plan, inputHash:'d'.repeat(64), logitsBits:'00000000', result:{value:'yes'}, rows, totalInstructions:rows.length*100, maxInstructions:100};
const actual = {...baseline, module};
const raw = [baseline, actual].map(x => Buffer.from(JSON.stringify(x)));
const recorded = {n:1, reference:baseline.totalInstructions, candidate:actual.totalInstructions, evidence:raw.map(sha)};
validatePair(raw, 1, release.module_hash, module, recorded);
assert.throws(() => validatePair(raw, 1, release.module_hash, module, {...recorded, evidence:undefined}), /Evidence bytes changed/);
const changed = JSON.parse(raw[1]); changed.result.value = 'no';
const tampered = [raw[0], Buffer.from(JSON.stringify(changed))];
assert.throws(() => validatePair(tampered, 1, release.module_hash, module, recorded), /Evidence bytes changed/);
assert.throws(() => validatePair(tampered, 1, release.module_hash, module, {...recorded, evidence:tampered.map(sha)}));
const overBudget = JSON.parse(raw[1]); overBudget.rows[0].instructions=4_000_000_001;
overBudget.totalInstructions=overBudget.rows.reduce((sum, row) => sum+row.instructions, 0); overBudget.maxInstructions=4_000_000_001;
const budgetRaw=[raw[0],Buffer.from(JSON.stringify(overBudget))];
assert.throws(() => validatePair(budgetRaw, 1, release.module_hash, module, {...recorded, evidence:budgetRaw.map(sha)}));
assert.throws(() => validatePair(raw, 1, release.module_hash, module, {...recorded, candidate:1}));
await assert.rejects(loadValidatedEvidence('unused', {complete:true, query_gate_passed:true,
 candidate_module:module, reference_module:release.module_hash,
 inputs:Array.from({length:91}, () => recorded)}, module));
console.log('Query evidence rejects changed bytes, numerical mismatches, budget violations and stale counts.');
