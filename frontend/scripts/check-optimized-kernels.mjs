/** Model-free numerical checks of the actual patched verification Wasm. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const path = process.argv[2];
assert(path, 'Pass the kernel-contract-tests optimized.wasm');
const module = new WebAssembly.Module(await readFile(path));
const imports = {};
for (const item of WebAssembly.Module.imports(module)) {
  assert.equal(item.kind, 'function');
  (imports[item.module] ??= {})[item.name] = () => { throw new Error(`Unexpected IC import: ${item.name}`); };
}
const { exports: e } = new WebAssembly.Instance(module, imports);
for (const name of ['check_integer', 'check_float', 'check_delta', 'check_predicates', 'check_elementwise', 'check_attention_scores']) assert.equal(typeof e[name], 'function', 'A verification-only build is required');
let integer = 0, float = 0, delta = 0, predicates = 0, elementwise = 0, attention = 0;
for (const n of [1, 7, 8, 45, 48, 57, 64, 87, 89, 91]) {
  for (const rows of [32, 64, 96, 128, 160, 168, 256, 512]) {
    for (const cols of [256, 512]) for (const pattern of [0, 1, 2]) {
      assert.equal(e.check_integer(n, rows, cols, pattern), 0, `INT8 bit mismatch: ${n}/${rows}/${cols}/${pattern}`);
      integer++;
    }
  }
}
for (const n of [1, 7, 8, 45, 91]) for (const rows of [32, 64, 128, 512]) for (const cols of [64, 128, 512]) {
  assert.equal(e.check_float(n, rows, cols), 0, `F32 bit mismatch: ${n}/${rows}/${cols}`);
  float++;
}
for (const n of [1, 2, 5, 7, 8, 45, 91]) {
  assert.equal(e.check_delta(n), 0, `Delta bit mismatch: ${n}`);
  delta++;
}
for (const n of [0, 1, 3, 4, 63, 64, 65, 127, 128, 129, 512]) for (let pattern = 0; pattern < 10; pattern++) {
  assert.equal(e.check_predicates(n, pattern), 0, `Predicate/RMS mismatch: ${n}/${pattern}`);
  predicates++;
}
for (const n of [1, 3, 7, 16, 64, 89, 91]) for (const c of [3, 4, 7, 128, 129]) {
  assert.equal(e.check_elementwise(n,c), 0, `Elementwise SIMD mismatch: ${n}/${c}`);
  elementwise++;
}
for (const n of [1, 7, 16, 45, 89, 91]) for (const prefix of [0, 5]) {
  assert.equal(e.check_attention_scores(n, prefix), 0, `Attention score mismatch: ${n}/${prefix}`);
  attention++;
}
console.log(JSON.stringify({ complete: true, integer, float, delta, predicates, elementwise, attention, comparison: 'Scalar references; exact F32 bits; zero/extreme/signed inputs and seeded continuations', scope: 'Synthetic arithmetic only; not full model inference or instruction calibration' }));
