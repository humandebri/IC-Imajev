import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createPrefixLoader } from "../src/prefix-loader.ts";
import { prepareQueryRun } from "../src/prepare-query-run.ts";
import { sha } from "../src/query-codec.ts";
import release from "../src/inference-release.json" with { type: "json" };

const base = new URL("../public/inference/prefix27-v1/", import.meta.url);
const manifestBytes = new Uint8Array(await readFile(new URL("manifest.json", base)));
const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
const assets = await Promise.all(Array.from({ length: 32 }, (_, layer) => readFile(new URL(manifest.assets[layer].file, base))));
const totalBytes = assets.reduce((sum, bytes) => sum + bytes.length, 0);
const originalFetch = globalThis.fetch;
const requests = [], gates = [];
let hold = false, corrupt = false, manifestOverride;
globalThis.fetch = async (url, { signal } = {}) => {
  signal?.throwIfAborted();
  const layer = /layer-(\d+)\.bin$/.exec(String(url));
  const index = layer ? Number(layer[1]) : null;
  requests.push(index);
  if (hold && layer) await new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    gates.push({ index, signal, resolve() { signal.removeEventListener("abort", abort); resolve(); } });
  });
  signal?.throwIfAborted();
  const bytes = index === null ? manifestOverride ?? manifestBytes : assets[index];
  const body = new Uint8Array(bytes);
  if (corrupt && index === 0) body[0] ^= 1;
  return new Response(body);
};
const flush = async predicate => {
  for (let i = 0; i < 100 && !predicate(); i++) await new Promise(resolve => setTimeout(resolve, 2));
  assert(predicate(), "Pending operation did not settle");
};
try {
  // Only current/next fetches start, and prefetched data can be consumed without waiting for another fetch.
  const store = createPrefixLoader(release.manifest_sha256), controller = new AbortController();
  const events = [];
  const first = await store.load("https://fixture/", controller.signal, event => events.push(event));
  hold = true;
  const firstAsset = first.asset(0);
  assert.deepEqual(requests, [null, 0, 1]);
  assert.equal(store.cachedBytes(), 0);
  gates.find(g => g.index === 0).resolve();
  assert.deepEqual(await firstAsset, new Uint8Array(assets[0]));
  gates.find(g => g.index === 1).resolve();
  await flush(() => store.cachedBytes() === assets[0].length + assets[1].length);
  assert.deepEqual(await first.asset(1), new Uint8Array(assets[1]));
  assert.deepEqual(requests, [null, 0, 1, 2]);
  assert(events.some(e => e.phase === "prefix-cache" && e.layer === 1));
  controller.abort();

  // A full warm run performs no prefix fetch or hash and stays within the fixed byte budget.
  hold = false; requests.length = 0;
  const all = createPrefixLoader(release.manifest_sha256);
  const coldController = new AbortController();
  const cold = await all.load("https://fixture/", coldController.signal);
  for (let layer = 0; layer < 32; layer++) assert.deepEqual(await cold.asset(layer), new Uint8Array(assets[layer]));
  assert.equal(requests.length, 33);
  assert.equal(all.cachedBytes(), totalBytes);
  assert.equal(totalBytes, 15_418_368);
  const warmEvents = [];
  const warm = await all.load("https://fixture/", new AbortController().signal, e => warmEvents.push(e));
  for (let layer = 0; layer < 32; layer++) await warm.asset(layer);
  assert.equal(requests.length, 33);
  assert(!warmEvents.some(e => e.phase === "prefix-fetch" || e.phase === "prefix-hash"));
  assert.equal(warmEvents.filter(e => e.phase === "prefix-cache").length, 32);
  await all.load("https://other-origin/", new AbortController().signal);
  assert.equal(all.cachedBytes(), 0, "Different source must not reuse buffers");

  // Cancelling a run cannot poison or cancel the independently fetched rerun.
  const isolated = createPrefixLoader(release.manifest_sha256);
  const c1 = new AbortController(), c2 = new AbortController();
  const r1 = await isolated.load("https://isolation/", c1.signal);
  const r2 = await isolated.load("https://isolation/", c2.signal);
  hold = true;
  const p1 = r1.asset(0), p2 = r2.asset(0);
  c1.abort();
  await assert.rejects(p1, /abort/i);
  for (const gate of gates.filter(g => g.signal === c2.signal)) gate.resolve();
  assert.deepEqual(await p2, new Uint8Array(assets[0]));
  assert(!c2.signal.aborted);
  c2.abort();

  // Failed checks are not retained or automatically retried in the same run.
  hold = false; corrupt = true; requests.length = 0;
  const failures = createPrefixLoader(release.manifest_sha256), failedController = new AbortController();
  const bad = await failures.load("https://failures/", failedController.signal);
  await assert.rejects(bad.asset(0), /mismatch/);
  await assert.rejects(bad.asset(0), /mismatch/);
  assert.equal(requests.filter(layer => layer === 0).length, 1);
  failedController.abort(); corrupt = false;
  const retryController = new AbortController();
  const retry = await failures.load("https://failures/", retryController.signal);
  assert.deepEqual(await retry.asset(0), new Uint8Array(assets[0]));
  retryController.abort();
  const wrongManifest = createPrefixLoader("0".repeat(64));
  await assert.rejects(wrongManifest.load("https://wrong/", new AbortController().signal), /manifest mismatch/);
  assert.equal(wrongManifest.cachedBytes(), 0);
  const oversized = structuredClone(manifest);
  oversized.assets[0].bytes = 15_418_369;
  manifestOverride = new TextEncoder().encode(JSON.stringify(oversized));
  const tooLarge = createPrefixLoader(await sha(manifestOverride));
  await assert.rejects(tooLarge.load("https://large/", new AbortController().signal), /budget/);
  manifestOverride = undefined;

  // All preparation and readiness operations must start before any of their barriers resolve.
  const started = [], resolve = {};
  const deferred = name => { started.push(name); return new Promise(done => { resolve[name] = done; }); };
  const client = { query: name => deferred(name), checkModule: () => deferred("module") };
  const preparation = prepareQueryRun("https://prepare/", [...manifest.prefix, 1], ["yes", "no"], new AbortController().signal,
    undefined, { loadPrefix: () => deferred("prefix"), createQueryClient: () => deferred("agent") });
  assert.deepEqual(started, ["prefix", "agent"]);
  resolve.prefix({ manifest, asset: async () => assets[0] }); resolve.agent(client);
  await flush(() => started.length === 5);
  assert.deepEqual(started.slice(2), ["module", "pack_status", "weight_cache_status"]);
  resolve.module();
  resolve.pack_status({ ready: true, model: release.model, pack_hash: release.pack_hash, bytes: BigInt(manifest.model_bytes) });
  let completed = false; void preparation.then(() => { completed = true; });
  await new Promise(done => setTimeout(done, 0)); assert.equal(completed, false);
  resolve.weight_cache_status({ names: Array(721).fill("weight") });
  assert.equal((await preparation).client, client);
  let queries = 0;
  await assert.rejects(prepareQueryRun("https://prepare/", [...manifest.prefix, 1], ["__unknown__", "yes"], new AbortController().signal,
    undefined, { loadPrefix: async () => ({ manifest, asset: async () => assets[0] }),
      createQueryClient: async () => ({ checkModule: async () => {}, query: async () => { queries++; } }) }), /reserved/);
  assert.equal(queries, 0);
  await assert.rejects(prepareQueryRun("https://prepare/", [...manifest.prefix, 1], ["yes", "no"], new AbortController().signal,
    undefined, { loadPrefix: async () => ({ manifest, asset: async () => assets[0] }),
      createQueryClient: async () => ({ checkModule: async () => { throw new Error("Module mismatch"); }, query: async () => ({}) }) }), /Module mismatch/);
  console.log("Prefix lookahead, warm reuse, 15.4 MB budget, isolation/cancel/retry/corruption and parallel readiness barriers passed.");
} finally { globalThis.fetch = originalFetch; }
