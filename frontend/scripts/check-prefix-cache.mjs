import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadPrefix, prefixCacheBytes } from "../src/prefix-assets.ts";
import release from "../src/inference-release.json" with { type: "json" };

const root = new URL("../public/inference/prefix27-v1/", import.meta.url);
const manifestBytes = await readFile(new URL("manifest.json", root));
const manifest = JSON.parse(manifestBytes);
const files = new Map(await Promise.all(Object.values(manifest.assets).map(async entry => [entry.file, await readFile(new URL(entry.file, root))])));
const originalFetch = globalThis.fetch;
let requests = [], corrupt = false, delay = false, active = 0, peak = 0;
let respond;
globalThis.fetch = async (url, init) => {
  requests.push(String(url));
  const response = await respond?.(String(url), init);
  if (response) return response;
  if (String(url).endsWith("manifest.json")) return new Response(manifestBytes);
  active++; peak = Math.max(peak, active);
  try {
    if (delay) await new Promise((resolve, reject) => {
      if (init.signal.aborted) return reject(init.signal.reason);
      init.signal.addEventListener("abort", () => reject(init.signal.reason), { once: true });
    });
    // Leave requests in flight long enough to exercise coalescing and lookahead.
    await new Promise(resolve => setTimeout(resolve, 5));
    init.signal.throwIfAborted();
    const bytes = files.get(String(url).split("/").at(-1));
    const body = corrupt ? Buffer.alloc(bytes.length) : bytes;
    return new Response(body);
  } finally { active--; }
};
try {
  const base = `https://offline.test/inference/immutable/${release.manifest_sha256}/`;
  const signal = new AbortController().signal;
  const events = [];
  const prefix = await loadPrefix(base, signal, event => events.push(event));
  assert.equal(prefixCacheBytes(), 0, "unverified buffers are not retained");
  assert.equal(requests.length, 3, "manifest plus two speculative layers");
  const [a, b] = await Promise.all([prefix.asset(0), prefix.asset(0)]);
  assert.strictEqual(a, b, "one verified buffer per layer");
  assert.equal(requests.filter(url => url.endsWith("layer-00.bin")).length, 1);
  for (let layer = 0; layer < 32; layer++) {
    assert.deepEqual(Buffer.from(await prefix.asset(layer)), files.get(manifest.assets[layer].file));
  }
  assert.ok(peak <= 3, "at most current layer and two future layers for sequential inference");
  assert.equal(prefixCacheBytes(), 15_418_368);
  assert.equal(events.filter(e => e.phase === "prefix-fetch").length, 33);
  assert.equal(events.filter(e => e.phase === "prefix-hash").length, 33);
  const before = requests.length;
  const warmEvents = [];
  const warm = await loadPrefix(base, signal, event => warmEvents.push(event));
  for (let layer = 0; layer < 32; layer++) await warm.asset(layer);
  assert.equal(requests.length, before, "warm run has no manifest or asset fetches");
  assert(!warmEvents.some(e => e.phase === "prefix-fetch" || e.phase === "prefix-hash"));
  assert.equal(warmEvents.filter(e => e.phase === "prefix-cache").length, 32);

  // Independent in-flight requests share only completed buffers and byte accounting.
  const concurrentA = await loadPrefix(`${base}concurrent/`, signal);
  const concurrentB = await loadPrefix(`${base}concurrent/`, signal);
  for (let layer = 0; layer < 32; layer++) {
    const [first, second] = await Promise.all([concurrentA.asset(layer), concurrentB.asset(layer)]);
    assert.strictEqual(first, second);
    assert(prefixCacheBytes() <= 15_418_368);
  }
  assert.equal(prefixCacheBytes(), 15_418_368, "duplicate fetches cannot double-count retained bytes");

  corrupt = true;
  const bad = await loadPrefix(`${base}bad/`, signal);
  await assert.rejects(bad.asset(0), /Prefix state mismatch/);
  await Promise.allSettled([bad.asset(1), bad.asset(2)]);
  corrupt = false;
  const repaired = await loadPrefix(`${base}bad/`, signal);
  assert.deepEqual(Buffer.from(await repaired.asset(0)), files.get("layer-00.bin"));
  // Let this loader's small lookahead settle before measuring cancellation.
  await Promise.all([repaired.asset(1), repaired.asset(2)]);
  await Promise.all([repaired.asset(3), repaired.asset(4)]);

  delay = true;
  const cancelled = new AbortController();
  const aborted = await loadPrefix(`${base}cancel/`, cancelled.signal);
  delay = false;
  const independent = await loadPrefix(`${base}cancel/`, signal);
  cancelled.abort();
  await assert.rejects(aborted.asset(0), /abort/i);
  assert.deepEqual(Buffer.from(await independent.asset(0)), files.get("layer-00.bin"));
  const count = requests.length;
  await assert.rejects(loadPrefix(base, cancelled.signal), /abort/i);
  assert.equal(requests.length, count, "already cancelled run starts no fetches");

  // A failed lookahead must not be sticky when its layer is demanded later.
  let healthy = false, attempts = 0;
  respond = async url => {
    if (url.endsWith("layer-01.bin")) {
      attempts++;
      if (!healthy) return new Response("", { status: 503 });
    }
  };
  const recovered = await loadPrefix(`${base}transient/`, signal);
  await recovered.asset(0);
  healthy = true;
  assert.deepEqual(Buffer.from(await recovered.asset(1)), files.get("layer-01.bin"));
  assert.equal(attempts, 2, "failed speculative request retried once on demand");

  // Demand can arrive while the failing lookahead is still in flight.
  let releaseFailure;
  attempts = 0;
  respond = async url => {
    if (url.endsWith("layer-00.bin")) {
      attempts++;
      if (attempts === 1) return new Promise(resolve => { releaseFailure = () => resolve(new Response("", { status: 503 })); });
    }
  };
  const inFlight = await loadPrefix(`${base}inflight/`, signal);
  const demanded = inFlight.asset(0);
  releaseFailure();
  assert.deepEqual(Buffer.from(await demanded), files.get("layer-00.bin"));
  assert.equal(attempts, 2);

  // Cached invalid bytes get exactly one reload-mode recovery request.
  for (const filename of ["manifest.json", "layer-00.bin"]) {
    const modes = [];
    respond = async (url, init) => {
      if (!url.endsWith(filename)) return;
      modes.push(init.cache);
      if (init.cache !== "reload") return new Response("invalid cached bytes");
    };
    const healed = await loadPrefix(`${base}integrity-${filename}/`, signal);
    await healed.asset(0);
    assert.deepEqual(modes, ["default", "reload"]);
  }

  // Never loop if a cache bypass still returns invalid bytes.
  const modes = [];
  respond = async (url, init) => {
    if (url.endsWith("layer-00.bin")) {
      modes.push(init.cache);
      return new Response(Buffer.alloc(files.get("layer-00.bin").length));
    }
  };
  const permanentlyBad = await loadPrefix(`${base}permanent/`, signal);
  await assert.rejects(permanentlyBad.asset(0), /Prefix state mismatch/);
  assert.deepEqual(modes, ["default", "reload"]);

  // Cancellation during integrity checking cannot start a cache-bypass request.
  const duringRepair = new AbortController(), abortedModes = [];
  respond = async (url, init) => {
    if (url.endsWith("layer-00.bin")) {
      abortedModes.push(init.cache);
      queueMicrotask(() => duringRepair.abort());
      return new Response("invalid");
    }
  };
  const cancelledRepair = await loadPrefix(`${base}cancel-repair/`, duringRepair.signal);
  await assert.rejects(cancelledRepair.asset(0), /abort/i);
  assert.deepEqual(abortedModes, ["default"]);

  // A demand retry after failed lookahead is bounded even on a persistent 503.
  let busyAttempts = 0;
  respond = async url => {
    if (url.endsWith("layer-00.bin")) {
      busyAttempts++;
      return new Response("", { status: 503 });
    }
  };
  const busy = await loadPrefix(`${base}persistent-busy/`, signal);
  await assert.rejects(busy.asset(0), /Could not load prefix state/);
  assert.equal(busyAttempts, 2);
  respond = undefined;
  console.log("Verified prefix lookahead, warm reuse, transient/in-flight recovery, bounded cache-bypass integrity recovery and independent cancellation passed.");
} finally { globalThis.fetch = originalFetch; }
