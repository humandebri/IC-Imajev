import assert from 'node:assert/strict';

/** Check both browser URL layouts against the candidate's release pins. */
export async function validatePrefixAssets(release, checked) {
  assert(/^[0-9a-f]{64}$/.test(release.manifest_sha256), 'Invalid manifest checksum');
  for (const directory of ['inference/prefix5-v1', `inference/immutable/${release.manifest_sha256}`]) {
    const manifest = JSON.parse(await checked(`${directory}/manifest.json`, release.manifest_sha256));
    for (const key of ['module_hash', 'model', 'pack_hash', 'prompt_layout']) {
      assert.equal(manifest[key], release[key], `Prefix ${key} mismatch`);
    }
    assert.deepEqual(manifest.prefix, [248045, 846, 198, 1349, 25]);
    assert.equal(Object.keys(manifest.assets).length, 32);
    for (let layer = 0; layer < 32; layer++) {
      const asset = manifest.assets[String(layer)];
      assert.equal(asset.file, `layer-${String(layer).padStart(2, '0')}.bin`);
      assert(/^[0-9a-f]{64}$/.test(asset.sha256));
      assert(Number.isSafeInteger(asset.bytes) && asset.bytes > 0);
      await checked(`${directory}/${asset.file}`, asset.sha256, asset.bytes);
    }
  }
}
