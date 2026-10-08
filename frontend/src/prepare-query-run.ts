import { createQueryClient, loadPrefix } from "./inference-agent.ts";
import { validateInput } from "./query-runner.ts";
import release from "./inference-release.json" with { type: "json" };
import type { TimingObserver } from "./inference-diagnostics.ts";

/** No inference query is issued until all independent preflight checks succeed. */
export async function prepareQueryRun(base: string, ids: number[], options: string[], signal: AbortSignal,
  observe?: TimingObserver, dependencies = { loadPrefix, createQueryClient }) {
  const [prefix, client] = await Promise.all([
    dependencies.loadPrefix(base, signal, observe), dependencies.createQueryClient(signal, observe),
  ]);
  signal.throwIfAborted();
  validateInput(ids, options, prefix.manifest);
  const [, pack, cache] = await Promise.all([
    client.checkModule(),
    client.query("pack_status", []) as Promise<{ model: string; pack_hash: string; ready: boolean; bytes: bigint }>,
    client.query("weight_cache_status", []) as Promise<{ names: string[] }>,
  ]);
  signal.throwIfAborted();
  if (!pack.ready || pack.model !== release.model || pack.pack_hash !== release.pack_hash ||
      Number(pack.bytes) !== prefix.manifest.model_bytes || cache.names.length !== 721) {
    throw new Error("Inference canister is being prepared. Please try later.");
  }
  return { prefix, client };
}
