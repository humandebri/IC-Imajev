const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const directory = path.join(root, 'artifacts/sha256-four-lane-kernels-v1');
const fixtureDir = path.join(root, 'artifacts/sha256-batch-real-fixtures-v1');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const requireThat = (value, context) => { if (!value) throw Error(context); };
const build = JSON.parse(fs.readFileSync(path.join(directory, 'build.json')));
const fixtures = JSON.parse(fs.readFileSync(path.join(fixtureDir, 'report.json')));
const audit = JSON.parse(fs.readFileSync(path.join(fixtureDir, 'independent-bytes-audit.json')));
requireThat(audit.complete && audit.all_raw_hashes_exact, 'fixture audit');
requireThat(sha(fs.readFileSync(path.join(fixtureDir, 'report.json'))) === audit.fixture_report_sha256, 'fixture identity');
requireThat(sha(fs.readFileSync(build.constants_path)) === build.constants_sha256, 'SHA constants');
for (const owner of [build, fixtures]) {
  for (const [file, hash] of Object.entries(owner.source_hashes)) requireThat(sha(fs.readFileSync(path.join(root, file))) === hash, file);
}
const wasm = fs.readFileSync(path.join(directory, 'sha256-four.wasm'));
requireThat(sha(wasm) === build.wasm_sha256, 'Wasm identity');
const memory = new WebAssembly.Memory({ initial: 1 });
const instance = new WebAssembly.Instance(new WebAssembly.Module(wasm), { env: { memory } });
function padded(bytes) {
  const n = Math.ceil((bytes.length + 9) / 64) * 64;
  const out = Buffer.alloc(n); bytes.copy(out); out[bytes.length] = 128;
  out.writeBigUInt64BE(BigInt(bytes.length) * 8n, n - 8); return out;
}
const cases = fixtures.groups.map(group => ({
  name: `${group.case}/${group.kind}`,
  messages: group.messages.map(item => fs.readFileSync(path.join(root, item.path))),
  expected: group.messages.map(item => item.sha256)
}));
for (const length of [0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 255, 256, 257, 4095, 4096]) {
  const messages = Array.from({ length: 4 }, (_, lane) => {
    const bytes = Buffer.alloc(length);
    for (let i = 0; i < length; i++) bytes[i] = (i * 73 + lane * 53 + (i >>> 3)) & 255;
    return bytes;
  });
  cases.push({ name: `boundary/${length}`, messages, expected: messages.map(sha) });
}
const results = [];
for (const item of cases) {
  const inputs = item.messages.map(padded);
  requireThat(inputs.every(value => value.length === inputs[0].length), 'equal padded lengths');
  const pointers = []; let end = 64;
  for (const input of inputs) { pointers.push(end); end += input.length + 64; }
  const output = end; end += 192;
  if (end > memory.buffer.byteLength) memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
  const bytes = Buffer.from(memory.buffer); bytes.fill(165);
  inputs.forEach((input, lane) => input.copy(bytes, pointers[lane]));
  const expectedMemory = Buffer.from(bytes);
  for (let word = 0; word < 8; word++) {
    for (let lane = 0; lane < 4; lane++) {
      const expected = Buffer.from(item.expected[lane], 'hex');
      expectedMemory.writeUInt32LE(expected.readUInt32BE(word * 4), output + word * 16 + lane * 4);
    }
  }
  instance.exports.sha256_four(...pointers, inputs[0].length / 64, output);
  requireThat(bytes.equals(expectedMemory), `digest/read-only/sentinel mismatch: ${item.name}`);
  results.push({ name: item.name, raw_bytes: item.messages[0].length, blocks: inputs[0].length / 64, all_four_digests_exact: true, all_memory_bytes_exact: true });
}
requireThat(results.length === 28, '28 batch cases');
const files = [__filename, path.join(directory, 'build.json'), path.join(directory, 'sha256-four.wat'), path.join(directory, 'sha256-four.wasm'), path.join(fixtureDir, 'report.json'), path.join(fixtureDir, 'independent-bytes-audit.json')];
const hashes = Object.fromEntries(files.map(file => [path.relative(root, file), sha(fs.readFileSync(file))]));
fs.writeFileSync(path.join(directory, 'execution-report.json'), JSON.stringify({
  complete: true, batch_cases: 28, messages: 112, results, source_hashes: hashes,
  ic_performance_verified: false, adopted: false, full_paid_goal_achieved: false,
  scope: 'Actual ordinary SIMD four-lane SHA256 compression against independent Node crypto; all36 real messages plus76 padding-boundary messages. Entire memory equals expected output with immutable inputs and sentinels. Padding/buffer setup runs in Node; no IC or full inference timing claim.'
}, null, 2) + '\n');
console.log('PASS 112 exact digests / 28 complete-memory SIMD batch cases');
