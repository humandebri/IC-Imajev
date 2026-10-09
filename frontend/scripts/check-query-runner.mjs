import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { IDL } from "@icp-sdk/core/candid";
import { methods, loadPrefix } from "../src/inference-agent.ts";
import { runQueryGraph, validateInput } from "../src/query-runner.ts";
import { selectPlan, MAX_TOKENS, queryCount } from "../src/query-plan.ts";
import release from "../src/inference-release.json" with { type: "json" };
import { frame, unframe, sha } from "../src/query-codec.ts";
const root = new URL("../../", import.meta.url);
const json = async path => JSON.parse(await readFile(new URL(path, root), "utf8"));
const manifest = await json("frontend/public/inference/prefix5-v1/manifest.json");
const report = await json("artifacts/prefix5-local-20261009/measurements/boom-653.json");
const wires = await json("artifacts/prefix5-local-20261009/boom653-wire.json");
assert.equal(report.moduleHash, release.module_hash);
const record = { token_ids: report.token_ids, options: report.options };
const plan = selectPlan(report.n), total = queryCount(plan);
assert.deepEqual(plan, report.plan);
let count = 0;
const progress = [];
const controller = new AbortController();
const observed = [];
const client = {
  async query(method, args) {
    const i = count++, wire = wires[i];
    assert.equal(method, wire.method);
    assert.deepEqual(Buffer.from(IDL.encode(methods[method].args, args)), Buffer.from(wire.args, "base64"), `Candid request ${i}`);
    return IDL.decode(methods[method].reply, Buffer.from(wire.reply, "base64"))[0];
  },
  async checkModule() {},
};
const assets = async layer => new Uint8Array(await readFile(new URL(`frontend/public/inference/prefix5-v1/${manifest.assets[layer].file}`, root)));
const result = await runQueryGraph(record.token_ids, record.options, {
  manifest, asset: assets, client, plan: plan, signal: controller.signal, progress: n => progress.push(n),
  observe(i, request, reply) {
    const wire = wires[i];
    const expectedRequest = IDL.decode(methods[wire.method].args, Buffer.from(wire.args, "base64"))[0];
    const value = IDL.decode(methods[wire.method].reply, Buffer.from(wire.reply, "base64"))[0];
    assert.deepEqual(Buffer.from(request), Buffer.from(expectedRequest), `frame ${i}`);
    assert.deepEqual(Buffer.from(reply.subarray(0, -32)), Buffer.from((value.Ok.measurement ?? value.Ok).state).subarray(0, -32), `reply ${i}`);
    const size = new DataView(reply.buffer, reply.byteOffset, 4).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(reply.subarray(4, 4 + size)));
    observed.push(unframe(reply, header));
  },
});
await Promise.all(observed);
assert.equal(count, total);
assert.deepEqual(progress, Array.from({ length: total + 1 }, (_, i) => i));
for (const field of ["value", "probabilities", "unknown_probability", "abstained"]) assert.deepEqual(result[field], report.result[field]);
// Replay the same signed evidence with the actual prefetch/cache loader too.
const originalFetch = globalThis.fetch, prefetchController = new AbortController();
try {
  globalThis.fetch = async (url, init) => {
    init.signal.throwIfAborted();
    const name = new URL(url).pathname.split("/").at(-1);
    return new Response(await readFile(new URL(`frontend/public/inference/prefix5-v1/${name}`, root)));
  };
  const prefix = await loadPrefix("https://verified-fixture/", prefetchController.signal);
  count = 0;
  const prefetchedResult = await runQueryGraph(record.token_ids, record.options, {
    ...prefix, client, plan: plan, signal: prefetchController.signal, progress() {},
  });
  assert.equal(count, total);
  assert.deepEqual(prefetchedResult, result, "Prefetch preserves Candid byte parity and decision bits");
} finally { prefetchController.abort(); globalThis.fetch = originalFetch; }
assert.throws(() => validateInput([...manifest.prefix, ...Array(MAX_TOKENS - 4).fill(1)], record.options, manifest), /total maximum/);
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
  manifest, asset: assets, client, plan: plan, signal: midway.signal, progress(n) { if (n === 1) midway.abort(); },
}), /abort/i);
assert.equal(count, 1);

const firstValue = IDL.decode(methods[wires[0].method].reply, Buffer.from(wires[0].reply, "base64"))[0];
const firstReply = new Uint8Array(firstValue.Ok.state);
const headerSize = new DataView(firstReply.buffer, firstReply.byteOffset, 4).getUint32(0, true);
const header = JSON.parse(new TextDecoder().decode(firstReply.subarray(4, 4 + headerSize)));
const checked = await frame(header, firstReply.subarray(4 + headerSize, -32));
const corrupt = checked.slice(); corrupt[4 + headerSize] ^= 1;
await assert.rejects(unframe(corrupt, header, true), /checksum/);
const zeroFooter = firstReply.slice(); zeroFooter.fill(0, zeroFooter.length - 32);
await assert.rejects(unframe(zeroFooter, header), /checksum/);
assert.deepEqual(zeroFooter, firstReply);
await unframe(zeroFooter, header, true);
assert.deepEqual(zeroFooter.subarray(0, -32), firstReply.subarray(0, -32));
assert.equal(Buffer.from(zeroFooter.subarray(-32)).toString("hex"), await sha(firstReply.subarray(0, -32)),
  "Verified native replies receive a checksum over their original serialized header and payload");
for (const field of ["input_hash", "model", "pack_hash", "step", "dims"]) {
  const modified = { ...header, [field]: field === "step" ? 99 : field === "dims" ? [1] : "wrong" };
  const mismatched = await frame(modified, firstReply.subarray(4 + headerSize, -32));
  await assert.rejects(unframe(mismatched, header, true), /mismatch/);
}
console.log("Measured Candid requests, frames, replies and decision bits match the five-token local proof; corrupted/bound replies, invalid input and cancellation passed.");
