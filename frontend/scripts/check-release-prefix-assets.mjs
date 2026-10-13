import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, lstat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { validatePrefixAssets } from './release-prefix-assets.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const root = await mkdtemp(join(tmpdir(), 'imajev-prefix-check-'));
try {
  const data = Buffer.from('prefix fixture');
  const release = { module_hash: 'a'.repeat(64), model: 'fixture', pack_hash: 'b'.repeat(64), prompt_layout: 'text-only-state-prefix5-v1' };
  const manifest = { ...release, prefix: [248045, 846, 198, 1349, 25], assets: Object.fromEntries(Array.from({ length: 32 }, (_, i) => [i, {
    file: `layer-${String(i).padStart(2, '0')}.bin`, bytes: data.length, sha256: sha(data),
  }])) };
  const encoded = Buffer.from(JSON.stringify(manifest));
  release.manifest_sha256 = sha(encoded);
  const directories = ['inference/prefix5-v1', `inference/immutable/${release.manifest_sha256}`];
  for (const directory of directories) {
    await mkdir(join(root, 'public', directory), { recursive: true });
    await writeFile(join(root, 'public', directory, 'manifest.json'), encoded);
    for (const asset of Object.values(manifest.assets)) await writeFile(join(root, 'public', directory, asset.file), data);
  }
  const checked = async (name, hash, size) => {
    const bytes = await readFile(join(root, 'public', name));
    assert.equal(sha(bytes), hash, `Prefix checksum mismatch: ${name}`);
    if (size !== undefined) assert.equal(bytes.length, size);
    return bytes;
  };
  await validatePrefixAssets(release, checked);
  await assert.rejects(validatePrefixAssets({ ...release, module_hash: 'c'.repeat(64) }, checked), /module_hash mismatch/);
  const manifestPath = join(root, 'public', directories[0], 'manifest.json');
  await writeFile(manifestPath, Buffer.from('changed manifest'));
  await assert.rejects(validatePrefixAssets(release, checked), /checksum mismatch/);
  await writeFile(manifestPath, encoded);
  await assert.rejects(validatePrefixAssets(release, async (name, hash, size) => checked(name, hash, size === undefined ? undefined : size + 1)));
  const asset = join(root, 'public', directories[1], 'layer-31.bin');
  await rm(asset);
  await assert.rejects(validatePrefixAssets(release, checked), /ENOENT/);
  await writeFile(asset, Buffer.from('changed prefix'));
  await assert.rejects(validatePrefixAssets(release, checked), /checksum mismatch/);

  // Exercise the real proposal entry point: reject corrupt assets before writing anything.
  await mkdir(join(root, 'src'));
  const files = {
    'src/inference-release.json': release,
    'prefix-origin.json': { module: release.module_hash },
    'report.json': { complete: true, query_gate_passed: true, inputs: Array(91).fill({}), candidate_module: release.module_hash, reference_module: 'b'.repeat(64) },
    'paid.json': { complete: true, paid_numerical_gate_passed: true, intermediate_state_gate_passed: true, paid_instruction_gate_passed: true, whole_message_accounting_complete: true, remaining: [], candidate_module: release.module_hash, reference_module: 'b'.repeat(64) },
    'contract.json': { live_fault_refund_passed: true, upgrade_gate_passed: true, candidate_module: release.module_hash },
  };
  for (const [name, value] of Object.entries(files)) await writeFile(join(root, name), JSON.stringify(value));
  const result = spawnSync(process.execPath, ['--experimental-strip-types', new URL('./prepare-optimized-release-proposal.mjs', import.meta.url).pathname,
    '--directory', root, '--query-evidence', root, '--paid-evidence', join(root, 'paid.json'), '--contract-evidence', join(root, 'contract.json')], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Prefix checksum mismatch/);
  for (const name of ['measurements', 'migration.json', 'src/query-profiles.json']) await assert.rejects(lstat(join(root, name)), /ENOENT/);
  // Older summaries must not approve migration using checkpoint-only counts.
  for (const accounting of [false, undefined]) {
    await writeFile(join(root, 'paid.json'), JSON.stringify({ ...files['paid.json'], whole_message_accounting_complete: accounting }));
    const rejected = spawnSync(process.execPath, ['--experimental-strip-types', new URL('./prepare-optimized-release-proposal.mjs', import.meta.url).pathname,
      '--directory', root, '--query-evidence', root, '--paid-evidence', join(root, 'paid.json'), '--contract-evidence', join(root, 'contract.json')], { encoding: 'utf8' });
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /Whole-message paid instruction accounting/);
    await assert.rejects(lstat(join(root, 'migration.json')), /ENOENT/);
  }
  await writeFile(join(root, 'paid.json'), JSON.stringify({ ...files['paid.json'], reference_module: 'c'.repeat(64) }));
  const wrongReference = spawnSync(process.execPath, ['--experimental-strip-types', new URL('./prepare-optimized-release-proposal.mjs', import.meta.url).pathname,
    '--directory', root, '--query-evidence', root, '--paid-evidence', join(root, 'paid.json'), '--contract-evidence', join(root, 'contract.json')], { encoding: 'utf8' });
  assert.notEqual(wrongReference.status, 0);
  assert.match(wrongReference.stderr, /same reference module/);
  await assert.rejects(lstat(join(root, 'migration.json')), /ENOENT/);
} finally {
  await rm(root, { recursive: true, force: true });
}
console.log('Prefix assets reject missing files, changed bytes and mismatched modules before proposal output.');
