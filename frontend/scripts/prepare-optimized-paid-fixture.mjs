/** Deterministic token-length fixtures for dedicated local paid comparison. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { Tokenizer } from '@huggingface/tokenizers';
import { tokenizeInput } from '../src/tokenization.ts';
const load = async path => JSON.parse(await readFile(new URL(path, import.meta.url)));
const tokenizer = new Tokenizer(await load('../public/tokenizer/tokenizer.json'), await load('../public/tokenizer/tokenizer_config.json'));
const codes = (await load('../public/tokenizer/decision_readout.json')).codes.map(x => x.code);
const example = (await load('../src/boom-examples.generated.json'))[1];
const ids = tokenizeInput(tokenizer, example.input, codes).tokenIds;
const prefix = (await load('../public/inference/prefix5-v1/manifest.json')).prefix;
assert.deepEqual(ids.slice(0, 5), prefix);
const cases = [9, 96, 128, 256, 512, 1024].map(total => {
 const extra = Array.from({length: Math.max(0, total - ids.length)}, (_, i) => [198, 220, 16][i % 3]);
 const tokens = total < ids.length ? [...ids.slice(0, total - 4), ...ids.slice(-4)] : [...ids.slice(0, -4), ...extra, ...ids.slice(-4)];
 assert.equal(tokens.length, total);
 return { total, request: {token_ids: tokens, options: example.input.options} };
});
assert(process.argv[2], 'Fresh fixture output path required');
await writeFile(process.argv[2], JSON.stringify({format: 1, cases}, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify({lengths: cases.map(x => x.total)}));
