/** Generate reviewable frontend pins from fresh, fully checked candidate evidence. */
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { profileDigest } from './query-calibration-policy.mjs';
import { loadValidatedEvidence } from './optimized-query-evidence.mjs';
import { validatePrefixAssets } from './release-prefix-assets.mjs';

const { values } = parseArgs({ options: Object.fromEntries(['directory', 'query-evidence', 'paid-evidence', 'contract-evidence'].map(name => [name, { type: 'string' }])) });
for (const name of ['directory', 'query-evidence', 'paid-evidence', 'contract-evidence']) assert(values[name], `--${name} required`);
const json = async path => JSON.parse(await readFile(path));
const query = await json(`${values['query-evidence']}/report.json`);
const paid = await json(values['paid-evidence']);
const contract = await json(values['contract-evidence']);
const release = await json(`${values.directory}/src/inference-release.json`);
const prefix = await json(`${values.directory}/prefix-origin.json`);
assert(query.complete && query.query_gate_passed && query.inputs.length === 91);
assert(paid.complete && paid.paid_numerical_gate_passed && paid.intermediate_state_gate_passed);
assert(paid.remaining.length === 0 || (paid.remaining.length === 1 && paid.remaining[0] === 'Live fault/refund and upgrade verification'),
 'Unresolved paid measurement gates');
assert(paid.paid_instruction_gate_passed === true && paid.whole_message_accounting_complete === true,
 'Whole-message paid instruction accounting and budgets must pass');
assert(contract.live_fault_refund_passed && contract.upgrade_gate_passed);
assert.equal(query.candidate_module, release.module_hash);
assert.equal(paid.candidate_module, release.module_hash);
assert.equal(paid.reference_module, query.reference_module, 'Paid and query evidence must use the same reference module');
assert.equal(contract.candidate_module, release.module_hash);
assert.equal(prefix.module, release.module_hash);
const plans = {}, evidence = {}, approvedReports = {}, approvedProfiles = {};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
await validatePrefixAssets(release, async (name, hash, size) => {
 const path = `${values.directory}/public/${name}`;
 assert((await lstat(path)).isFile(), `Regular prefix asset required: ${name}`);
 const bytes = await readFile(path);
 assert.equal(sha(bytes), hash, `Prefix checksum mismatch: ${name}`);
 if (size !== undefined) assert.equal(bytes.length, size, `Prefix size mismatch: ${name}`);
 return bytes;
});
execFileSync('python3', ['-B', fileURLToPath(new URL('../../scripts/check_paid_instruction_evidence.py', import.meta.url)),
 '--report', values['paid-evidence']], { stdio: 'inherit' });
execFileSync('python3', ['-B', fileURLToPath(new URL('../../scripts/verify_optimized_paid_contract.py', import.meta.url)),
 'check', '--report', values['contract-evidence'], '--paid-report', values['paid-evidence']], { stdio: 'inherit' });
// Validate every original reference/candidate pair before writing proposal files.
const validated = await loadValidatedEvidence(values['query-evidence'], query, release.module_hash);
await mkdir(`${values.directory}/measurements`);
for (let n = 1; n <= 91; n++) {
 const { bytes, actual } = validated[n-1];
 assert.equal(actual.module, release.module_hash); assert.equal(actual.n, n);
 assert(actual.maxInstructions <= 4_000_000_000);
 assert.equal(actual.totalInstructions, actual.rows.reduce((sum, row) => sum + row.instructions, 0));
 plans[n] = { fronts: actual.plan.fronts, completions: actual.plan.completions };
 evidence[n] = { queries: actual.rows.length, maxInstructions: actual.maxInstructions,
  maxRequestBytes: Math.max(...actual.rows.map(r => r.requestBytes)), maxReplyBytes: Math.max(...actual.rows.map(r => r.replyBytes)),
  reference: 'separate-reference-module', finalPayloadHash: actual.rows.at(-1).payloadHash, reportSHA256: sha(bytes) };
 approvedReports[n] = evidence[n].reportSHA256;
 approvedProfiles[n] = profileDigest(release.module_hash, n, plans[n], evidence[n]);
 await writeFile(`${values.directory}/measurements/n${n}-candidate.json`, bytes, { flag: 'wx' });
}
await writeFile(`${values.directory}/src/query-profiles.json`, JSON.stringify({ moduleHash: release.module_hash, maxSuffix: 91, targetInstructions: 4_000_000_000, plans, evidence }, null, 2) + '\n', { flag: 'wx' });
await mkdir(`${values.directory}/scripts`);
await writeFile(`${values.directory}/scripts/query-plan-decisions.json`, JSON.stringify({ moduleHash: release.module_hash, approvedReports, approvedProfiles, rejectedReports: {}, reason: 'Fresh candidate measurements passed exact reference-module comparison and all migration gates' }, null, 2) + '\n', { flag: 'wx' });
for (const name of ['query-profiles.json', 'inference-release.json']) {
 await writeFile(`${values.directory}/previous-${name}`, await readFile(new URL(`../src/${name}`, import.meta.url)), { flag: 'wx' }).catch(error => { if (name !== 'inference-release.json' || error.code !== 'EEXIST') throw error; });
}
await writeFile(`${values.directory}/migration.json`, JSON.stringify({ module: release.module_hash, validated_candidate: true, deployed: false,
 query_report_sha256: sha(await readFile(`${values['query-evidence']}/report.json`)),
 paid_report_sha256: sha(await readFile(values['paid-evidence'])), contract_report_sha256: sha(await readFile(values['contract-evidence'])) }, null, 2) + '\n', { flag: 'wx' });
