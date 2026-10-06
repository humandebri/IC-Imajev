// Node/V8 wall time is a host-engine measurement, not an ICP instruction count.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {performance} from 'node:perf_hooks';

const [imajevPath, ggmlPath, reportPath] = process.argv.slice(2);
const imports = {ic0:{performance_counter:()=>0n, msg_arg_data_size:()=>0, msg_arg_data_copy:()=>{}, msg_reply_data_append:()=>{}, msg_reply:()=>{}}};
const instantiate = async path => (await WebAssembly.instantiate(fs.readFileSync(path), imports)).instance;
const imajev = await instantiate(imajevPath);
const ggml = await instantiate(ggmlPath);
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const fnv64 = bytes => {
  let hash = 14695981039346656037n;
  for (const byte of bytes) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 1099511628211n);
  return hash.toString(16).padStart(16, '0');
};
const output = instance => {
  const {memory, output_ptr, output_len} = instance.exports;
  return Buffer.from(new Uint8Array(memory.buffer, output_ptr(), output_len() * 4));
};
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const rows = [];
for (const cols of [256, 2560]) {
  for (const tokens of [1, 7, 32, 87, 132]) {
    const setupMs = {};
    for (const [name, instance] of [['imajev', imajev], ['ggml', ggml]]) {
      const start = performance.now(); instance.exports.setup(tokens, 512, cols);
      setupMs[name] = performance.now() - start;
    }
    imajev.exports.run(); ggml.exports.run();
    const expected = output(ggml);
    if (!expected.equals(output(imajev))) throw new Error(`bit mismatch: ${tokens}/512/${cols}`);
    // Independently check the selected exact-integer fixture against scalar JS.
    const ref = new Int32Array(tokens * 512);
    const value = (i, seed) => (((Math.imul(i, 1664525) + seed) >>> 16) % 63) - 31;
    for (let t = 0; t < tokens; t++) for (let r = 0; r < 512; r++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) sum += (c % 32 === 0 ? 127 : value(t * cols + c, 1013904223)) * value(r * cols + c, 777);
      ref[t * 512 + r] = sum;
    }
    const floats = new Float32Array(expected.buffer, expected.byteOffset, expected.length / 4);
    for (let i = 0; i < ref.length; i++) if (floats[i] !== ref[i]) throw new Error(`scalar mismatch at ${i}`);
    // Warm both functions before collecting alternating batches.
    for (let i = 0; i < 12; i++) { imajev.exports.run(); ggml.exports.run(); }
    const samples = {imajev: [], ggml: []};
    const repeats = cols === 256 ? 10 : 3;
    for (let sample = 0; sample < 9; sample++) {
      const order = sample % 2 === 0 ? [['imajev', imajev], ['ggml', ggml]] : [['ggml', ggml], ['imajev', imajev]];
      for (const [name, instance] of order) {
        const start = performance.now();
        for (let repeat = 0; repeat < repeats; repeat++) instance.exports.run();
        samples[name].push((performance.now() - start) / repeats);
      }
    }
    if (!expected.equals(output(imajev)) || !expected.equals(output(ggml))) throw new Error('output changed during measurement');
    const medians = Object.fromEntries(Object.entries(samples).map(([name, values]) => [name, median(values)]));
    const row = {tokens, rows:512, cols, outputSha256:sha(expected), outputFnv64Hex:fnv64(expected), bitwiseEqual:true, scalarEqual:true, setupMs, repeats, samplesMs:samples, medianMs:medians, imajevOverGgml:medians.imajev/medians.ggml};
    rows.push(row);
    console.log(JSON.stringify({tokens, cols, medianMs:medians, ratio:row.imajevOverGgml}));
  }
}
fs.writeFileSync(reportPath, JSON.stringify({scope:'Prepared synthetic projection in Node/V8; not full-model or ICP performance', node:process.version, nodeFlags:process.execArgv, warmups:12, samples:9, wasmHashes:{imajev:sha(fs.readFileSync(imajevPath)), ggml:sha(fs.readFileSync(ggmlPath))}, rows}, null, 2)+'\n');
