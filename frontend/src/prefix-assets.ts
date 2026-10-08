import release from "./inference-release.json" with { type: "json" };
import { sha } from "./query-codec.ts";
import { timed, type TimingObserver } from "./inference-diagnostics.ts";

export interface PrefixManifest {
  model: string; pack_hash: string; model_bytes: number; prefix: number[];
  assets: Record<string, { file: string; bytes: number; sha256: string }>;
}

// Retain only verified assets for one release/base (about 15.4 MB per worker).
// In-flight requests belong to a run, so cancelling one run cannot poison another.
const MAX_PREFIX_BYTES = 15_418_368;
let cached: { base: string; manifest: PrefixManifest; assets: Map<number, Uint8Array>; bytes: number } | undefined;
export const prefixCacheBytes = () => cached?.bytes ?? 0;
class PrefixIntegrityError extends Error {}

async function fetchVerified(url: string, signal: AbortSignal,
  valid: (bytes: Uint8Array) => Promise<boolean>, loadError: string, mismatchError: string,
  observe?: TimingObserver, detail: { layer?: number; bytes?: number } = {}) {
  // Retry integrity failures once with HTTP-cache bypass, and update that cache.
  // An origin that still serves invalid bytes must fail closed, without a loop.
  for (const cache of ["default", "reload"] as const) {
    signal.throwIfAborted();
    const bytes = await timed(observe, "prefix-fetch", async () => {
      const response = await fetch(url, {
        cache, signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
      });
      if (!response.ok) throw new Error(loadError);
      return new Uint8Array(await response.arrayBuffer());
    }, detail);
    const matches = await timed(observe, "prefix-hash", () => valid(bytes), detail);
    signal.throwIfAborted();
    if (matches) return bytes;
  }
  throw new PrefixIntegrityError(mismatchError);
}

export async function loadPrefix(base: string, signal: AbortSignal, observe?: TimingObserver) {
  signal.throwIfAborted();
  let store = cached?.base === base ? cached : undefined;
  if (!store) {
    const bytes = await fetchVerified(`${base}manifest.json`, signal,
      async bytes => await sha(bytes) === release.manifest_sha256,
      "Could not load prefix settings.", "Prefix manifest mismatch.", observe);
    const manifest = JSON.parse(new TextDecoder().decode(bytes)) as PrefixManifest;
    const entries = Array.from({ length: 32 }, (_, layer) => manifest.assets[String(layer)]);
    if (Object.keys(manifest.assets).length !== 32 || entries.some(entry => !entry ||
        !Number.isSafeInteger(entry.bytes) || entry.bytes <= 0) ||
        entries.reduce((sum, entry) => sum + entry.bytes, 0) > MAX_PREFIX_BYTES) {
      throw new Error("Invalid prefix asset budget.");
    }
    // Another run may have loaded this same verified manifest while we fetched it.
    store = cached?.base === base ? cached : {
      base, manifest, assets: new Map<number, Uint8Array>(), bytes: 0,
    };
    cached = store;
  }
  const verified = store;
  const pending = new Map<number, { promise: Promise<Uint8Array>; speculative: boolean }>();
  function load(layer: number, speculative: boolean) {
    signal.throwIfAborted();
    const bytes = verified.assets.get(layer);
    if (bytes) return { promise: Promise.resolve(bytes), speculative: false };
    const existing = pending.get(layer);
    if (existing) return existing;
    const promise = (async () => {
      const entry = verified.manifest.assets[String(layer)];
      if (!entry) throw new Error("Invalid prefix layer.");
      const bytes = await fetchVerified(`${base}${entry.file}`, signal,
        async bytes => bytes.length === entry.bytes && await sha(bytes) === entry.sha256,
        "Could not load prefix state.", "Prefix state mismatch.", observe, { layer, bytes: entry.bytes });
      const shared = verified.assets.get(layer);
      if (shared) return shared;
      if (verified.bytes + bytes.length > MAX_PREFIX_BYTES) throw new Error("Prefix cache budget exceeded.");
      if (cached === verified) {
        verified.assets.set(layer, bytes);
        verified.bytes += bytes.length;
      }
      return bytes;
    })();
    const request = { promise, speculative };
    pending.set(layer, request);
    // Failed lookahead never poisons a later demand. Keep rejections handled even
    // if this layer is never consumed because the inference fails or is cancelled.
    void promise.catch(() => {
      if (pending.get(layer) === request) pending.delete(layer);
    });
    return request;
  }
  function prefetch(first: number) {
    for (let layer = first; layer < first + 2; layer++) {
      if (verified.manifest.assets[String(layer)]) load(layer, true);
    }
  }
  // Start while the caller checks module identity/readiness, before the first query.
  prefetch(0);
  return { manifest: verified.manifest, async asset(layer: number) {
    signal.throwIfAborted();
    const hit = verified.assets.has(layer);
    const current = load(layer, false);
    prefetch(layer + 1);
    let bytes: Uint8Array;
    try { bytes = await current.promise; }
    catch (error) {
      // Also recover when demand arrives before a speculative request fails.
      if (!current.speculative || signal.aborted || error instanceof PrefixIntegrityError) throw error;
      bytes = await load(layer, false).promise;
    }
    signal.throwIfAborted();
    if (hit) observe?.({ phase: "prefix-cache", durationMs: 0, layer, bytes: bytes.length });
    return bytes;
  } };
}
