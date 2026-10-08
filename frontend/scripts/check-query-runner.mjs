import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { IDL } from "@icp-sdk/core/candid";
import { methods, loadPrefix } from "../src/inference-agent.ts";
import { runQueryGraph, validateInput } from "../src/query-runner.ts";
import { baselinePlan, MAX_TOKENS } from "../src/query-plan.ts";
import { frame, unframe } from "../src/query-codec.ts";
const root = new URL("../../", import.meta.url);
const json = async path => JSON.parse(await readFile(new URL(path, root), "utf8"));
const manifest = await json("frontend/public/inference/prefix27-v1/manifest.json");
const record = (await json("artifacts/text-short-v2/inputs.json")).records[2];
const base = "artifacts/mainnet-prefix27-upgrade-20261007/anonymous-query-653/queries/";
let count = 0;
const progress = [];
const controller = new AbortController();
const observed = [];
const client = {
  async query(method, args) {
    const i = count++, stem = String(i).padStart(6, "0");
    const expected = await readFile(new URL(`${base}${stem}.args.bin`, root));
    assert.deepEqual(Buffer.from(IDL.encode(methods[method].args, args)), expected, `Candid request ${i}`);
    const hex = (await readFile(new URL(`${base}${stem}.reply.hex`, root), "utf8")).trim().replace(/^0x/, "");
    return IDL.decode(methods[method].reply, Buffer.from(hex, "hex"))[0];
  },
  async checkModule() {},
};
const assets = async layer => new Uint8Array(await readFile(new URL(`frontend/public/inference/prefix27-v1/${manifest.assets[layer].file}`, root)));
const result = await runQueryGraph(record.token_ids, record.options, {
  manifest, asset: assets, client, plan: baselinePlan(57), signal: controller.signal, progress: n => progress.push(n),
  observe(i, request, reply) {
    observed.push((async () => {
      const stem = String(i).padStart(6, "0");
      assert.deepEqual(Buffer.from(request), await readFile(new URL(`${base}${stem}.request.bin`, root)), `frame ${i}`);
      assert.deepEqual(Buffer.from(reply), await readFile(new URL(`${base}${stem}.response.bin`, root)), `sealed reply ${i}`);
    })());
  },
});
await Promise.all(observed);
assert.equal(count, 32);
assert.deepEqual(progress, Array.from({ length: 33 }, (_, i) => i));
const report = await json("artifacts/mainnet-prefix27-upgrade-20261007/anonymous-query-653/report.json");
for (const field of ["value", "probabilities", "unknown_probability", "abstained"]) assert.deepEqual(result[field], report.decision[field]);
// Replay the same signed evidence with the actual prefetch/cache loader too.
const originalFetch = globalThis.fetch, prefetchController = new AbortController();
try {
  globalThis.fetch = async (url, init) => {
    init.signal.throwIfAborted();
    const name = new URL(url).pathname.split("/").at(-1);
    return new Response(await readFile(new URL(`frontend/public/inference/prefix27-v1/${name}`, root)));
  };
  const prefix = await loadPrefix("https://verified-fixture/", prefetchController.signal);
  count = 0;
  const prefetchedResult = await runQueryGraph(record.token_ids, record.options, {
    ...prefix, client, plan: baselinePlan(57), signal: prefetchController.signal, progress() {},
  });
  assert.equal(count, 32);
  assert.deepEqual(prefetchedResult, result, "Prefetch preserves Candid byte parity and decision bits");
} finally { prefetchController.abort(); globalThis.fetch = originalFetch; }
assert.throws(() => validateInput([...manifest.prefix, ...Array(MAX_TOKENS - 26).fill(1)], record.options, manifest), /total maximum/);
assert.throws(() => validateInput([1, ...record.token_ids.slice(1)], record.options, manifest), /prefix/);
const cancelled = new AbortController(); cancelled.abort();
let calls = 0;
await assert.rejects(runQueryGraph(record.token_ids, record.options, {
  manifest, asset: assets, client: { query: async () => { calls++; }, checkModule: async () => {} },
  signal: cancelled.signal, progress() {},
}), /abort/i);
assert.equal(calls, 0);
// The reserved decision label must fail before any prefix fetch or query.
for (const reserved of ["__unknown__", "  __unknown__  "]) {
  let queryCalls = 0, assetCalls = 0;
  await assert.rejects(runQueryGraph(record.token_ids, [reserved, "yes"], {
    manifest, signal: controller.signal, progress() {},
    asset: async () => { assetCalls++; throw new Error("Unexpected prefix fetch"); },
    client: { query: async () => { queryCalls++; }, checkModule: async () => {} },
  }), /reserved for abstention/);
  assert.equal(queryCalls, 0);
  assert.equal(assetCalls, 0);
}
// Cancellation after a completed query cannot issue the next dependent call.
count = 0;
const midway = new AbortController();
await assert.rejects(runQueryGraph(record.token_ids, record.options, {
  manifest, asset: assets, client, plan: baselinePlan(57), signal: midway.signal, progress(n) { if (n === 1) midway.abort(); },
}), /abort/i);
assert.equal(count, 1);

const firstReply = new Uint8Array(await readFile(new URL(`${base}000000.response.bin`, root)));
const headerSize = new DataView(firstReply.buffer, firstReply.byteOffset, 4).getUint32(0, true);
const header = JSON.parse(new TextDecoder().decode(firstReply.subarray(4, 4 + headerSize)));
const corrupt = firstReply.slice(); corrupt[4 + headerSize] ^= 1;
await assert.rejects(unframe(corrupt, header, true), /checksum/);
const zeroFooter = firstReply.slice(); zeroFooter.fill(0, zeroFooter.length - 32);
await assert.rejects(unframe(zeroFooter, header), /checksum/);
await unframe(zeroFooter, header, true);
assert.deepEqual(zeroFooter, firstReply);
for (const field of ["input_hash", "model", "pack_hash", "step", "dims"]) {
  const modified = { ...header, [field]: field === "step" ? 99 : field === "dims" ? [1] : "wrong" };
  const mismatched = await frame(modified, firstReply.subarray(4 + headerSize, -32));
  await assert.rejects(unframe(mismatched, header, true), /mismatch/);
}
console.log("32 Candid requests, frames, sealed replies and decision bits match mainnet; corrupted/bound replies, invalid input and cancellation passed.");
