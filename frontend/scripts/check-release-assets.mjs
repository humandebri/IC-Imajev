import assert from 'node:assert/strict';
import { readFile, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { validatePrefixAssets } from './release-prefix-assets.mjs';

const root = new URL('../../', import.meta.url);
const publicRoot = new URL('../public/', import.meta.url);
const release = JSON.parse(await readFile(new URL('../src/inference-release.json', import.meta.url)));
const lock = JSON.parse(await readFile(new URL('MODEL_LOCK.json', root)));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const files = [];
async function checked(name, hash, size) {
  assert(!name.includes('..') && !name.startsWith('/') && !name.includes('\\'));
  const path = new URL(name, publicRoot);
  assert((await lstat(path)).isFile(), `Regular release asset required: ${name}`);
  const bytes = await readFile(path);
  assert.equal(sha(bytes), hash, `Release checksum mismatch: ${name}`);
  if (size !== undefined) assert.equal(bytes.length, size, `Release size mismatch: ${name}`);
  files.push(`frontend/public/${name}`);
  return bytes;
}
try {
  for (const [component, name] of [['base', 'tokenizer.json'], ['base', 'tokenizer_config.json'], ['adapter', 'decision_readout.json']]) {
    const file = lock.components[component].files[name];
    await checked(`tokenizer/${name}`, file.sha256, file.bytes);
  }
  const hashes = Object.fromEntries(files.map(path => {
    const name = path.split('/').at(-1);
    const component = name === 'decision_readout.json' ? 'adapter' : 'base';
    return [name, lock.components[component].files[name].sha256];
  }));
  const tokenizerManifest = new URL('tokenizer/manifest.json', publicRoot);
  assert((await lstat(tokenizerManifest)).isFile());
  assert.deepEqual(JSON.parse(await readFile(tokenizerManifest)), { hashes });
  files.push('frontend/public/tokenizer/manifest.json');
  await validatePrefixAssets(release, checked);
  if (process.argv.includes('--files')) console.log(JSON.stringify(files));
  else console.log(`Verified ${files.length} pinned browser release assets; no model weights required.`);
} catch (error) {
  console.error('Browser release assets are missing or invalid. Extract the matching public source release bundle, or explicitly regenerate assets using the retained model/prefix inputs. See docs/RELEASING.md.');
  throw error;
}
