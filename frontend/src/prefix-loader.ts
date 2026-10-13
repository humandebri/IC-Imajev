import release from "./inference-release.json" with { type: "json" };
import { sha } from "./query-codec.ts";
import { timed, type TimingObserver } from "./inference-diagnostics.ts";
import type { PrefixManifest } from "./inference-agent.ts";

const MAX_PREFIX_BYTES = 3_816_448;
type Record = { key: string; manifest: PrefixManifest; assets: Map<number, Uint8Array>; bytes: number };

/** One verified release per Worker. Only completed data is shared across runs. */
export function createPrefixLoader(manifestHash: string) {
  let cached: Record | undefined;
  return {
    cachedBytes: () => cached?.bytes ?? 0,
    async load(base: string, signal: AbortSignal, observe?: TimingObserver) {
      signal.throwIfAborted();
      const key = `${base}:${manifestHash}`;
      let record = cached?.key === key ? cached : undefined;
      if (!record) {
        const bytes = await timed(observe, "prefix-fetch", async () => {
          const response = await fetch(`${base}manifest.json`, { signal });
          if (!response.ok) throw new Error("Could not load prefix settings.");
          return new Uint8Array(await response.arrayBuffer());
        });
        if (await timed(observe, "prefix-hash", () => sha(bytes)) !== manifestHash) throw new Error("Prefix manifest mismatch.");
        signal.throwIfAborted();
        const manifest = JSON.parse(new TextDecoder().decode(bytes)) as PrefixManifest;
        if (manifest.module_hash !== release.module_hash || manifest.prompt_layout !== release.prompt_layout ||
            manifest.prefix.length !== 5 || !manifest.prefix.every((id, i) => id === [248045, 846, 198, 1349, 25][i])) {
          throw new Error("Prefix runtime or prompt layout mismatch.");
        }
        const entries = Array.from({ length: 32 }, (_, layer) => manifest.assets[String(layer)]);
        if (Object.keys(manifest.assets).length !== 32 || entries.some(entry => !entry ||
            !Number.isSafeInteger(entry.bytes) || entry.bytes <= 0) ||
            entries.reduce((sum, entry) => sum + entry.bytes, 0) > MAX_PREFIX_BYTES) {
          throw new Error("Invalid prefix asset budget.");
        }
        // Another run may have completed this same manifest in the meantime.
        record = cached?.key === key ? cached : { key, manifest, assets: new Map(), bytes: 0 };
        cached = record;
      }
      const verified = record;
      // Pending promises retain this run's signal and are never shared with a rerun.
      const pending = new Map<number, Promise<Uint8Array>>();
      const start = (layer: number): Promise<Uint8Array> => {
        const hit = verified.assets.get(layer);
        if (hit) return Promise.resolve(hit);
        const existing = pending.get(layer);
        if (existing) return existing;
        const promise = (async () => {
          signal.throwIfAborted();
          const entry = verified.manifest.assets[String(layer)];
          if (!entry) throw new Error("Invalid prefix layer.");
          const bytes = await timed(observe, "prefix-fetch", async () => {
            const response = await fetch(`${base}${entry.file}`, { signal });
            if (!response.ok) throw new Error("Could not load prefix state.");
            return new Uint8Array(await response.arrayBuffer());
          }, { layer, bytes: entry.bytes });
          if (bytes.length !== entry.bytes || await timed(observe, "prefix-hash", () => sha(bytes), { layer }) !== entry.sha256) {
            throw new Error("Prefix state mismatch.");
          }
          signal.throwIfAborted();
          if (cached === verified) {
            // Concurrent runs can fetch the same layer; keep only one verified buffer.
            const shared = verified.assets.get(layer);
            if (shared) return shared;
            if (verified.bytes + bytes.length > MAX_PREFIX_BYTES) throw new Error("Prefix cache budget exceeded.");
            verified.assets.set(layer, bytes);
            verified.bytes += bytes.length;
          }
          return bytes;
        })();
        pending.set(layer, promise);
        // A speculative failure is reported when that layer is needed, never unhandled.
        void promise.catch(() => {});
        return promise;
      };
      return { manifest: verified.manifest, async asset(layer: number) {
        signal.throwIfAborted();
        const hit = verified.assets.has(layer);
        const needed = start(layer);
        // The graph requests layers sequentially: overlap only the next layer.
        if (verified.manifest.assets[String(layer + 1)]) void start(layer + 1);
        const bytes = await needed;
        signal.throwIfAborted();
        if (hit) observe?.({ phase: "prefix-cache", durationMs: 0, layer, bytes: bytes.length });
        return bytes;
      } };
    },
  };
}
