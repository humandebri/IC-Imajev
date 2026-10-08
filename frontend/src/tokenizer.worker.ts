import { Tokenizer } from "@huggingface/tokenizers";
import { tokenizeInput } from "./tokenization.ts";
import type { DecisionInput } from "./types";
import { prefixCacheBytes } from "./inference-agent.ts";
import { prepareQueryRun } from "./prepare-query-run.ts";
import { runQueryGraph } from "./query-runner.ts";
import { MAX_TOKENS } from "./query-plan.ts";
import { timed, type TimingObserver, type TimingEvent } from "./inference-diagnostics.ts";
import release from "./inference-release.json" with { type: "json" };

async function loadTokenizer() {
  const base = `${import.meta.env.BASE_URL}tokenizer/`;
  const manifestResponse = await fetch(`${base}manifest.json`);
  if (!manifestResponse.ok)
    throw new Error("Could not load tokenizer settings.");
  const manifest = (await manifestResponse.json()) as {
    hashes: Record<string, string>;
  };
  async function load(name: string) {
    const response = await fetch(`${base}${name}`);
    if (!response.ok)
      throw new Error("Could not load a tokenizer file.");
    const buffer = await response.arrayBuffer();
    const hash = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
      (n) => n.toString(16).padStart(2, "0"),
    ).join("");
    if (hash !== manifest.hashes[name])
      throw new Error("Tokenizer file verification failed.");
    return JSON.parse(new TextDecoder().decode(buffer));
  }
  const [json, config, readout] = await Promise.all([
    load("tokenizer.json"),
    load("tokenizer_config.json"),
    load("decision_readout.json"),
  ]);
  return {
    tokenizer: new Tokenizer(json, config),
    codes: readout.codes.map(
      (entry: { code: string }) => entry.code,
    ) as string[],
  };
}
let loaded: ReturnType<typeof loadTokenizer> | undefined;
const runs = new Map<number, AbortController>();
self.onmessage = async (
  event: MessageEvent<{ id: number; input: DecisionInput; op?: "count" | "run" | "cancel"; diagnostics?: boolean }>,
) => {
  const { id, input, op } = event.data;
  if (op === "cancel") { runs.get(id)?.abort(); return; }
  const controller = op === "run" ? new AbortController() : undefined;
  if (controller) runs.set(id, controller);
  const started = performance.now();
  const events: TimingEvent[] = [];
  const observe: TimingObserver | undefined = event.data.diagnostics ? timing => events.push(timing) : undefined;
  try {
    if (!input || typeof input.state !== "string" || typeof input.question !== "string" ||
        input.state.length + input.question.length > 100_000 || !Array.isArray(input.options) ||
        input.options.length > 7 || input.options.some(o => typeof o !== "string" || o.length > 128)) {
      throw new Error("Invalid or oversized input.");
    }
    const { tokenizer, codes } = await (loaded ??= loadTokenizer());
    const tokenized = tokenizeInput(tokenizer, input, codes);
    if (!controller) { self.postMessage({ id, counts: tokenized.counts }); return; }
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(300_000)]);
    signal.throwIfAborted();
    if (!input.question.trim()) throw new Error("Enter a question.");
    if (tokenized.counts.total > MAX_TOKENS) throw new Error(`Input exceeds ${MAX_TOKENS} tokens. Shorten the context, question or options.`);
    const { prefix, client } = await timed(observe, "preparation", () =>
      prepareQueryRun(`${import.meta.env.BASE_URL}inference/immutable/${release.manifest_sha256}/`, tokenized.tokenIds, input.options, signal, observe));
    const result = await runQueryGraph(tokenized.tokenIds, input.options, {
      ...prefix, asset: layer => timed(observe, "prefix-wait", () => prefix.asset(layer), { layer }),
      client, signal, progress: (completed, total) => self.postMessage({ id, completed, total }),
    });
    await client.checkModule();
    signal.throwIfAborted();
    observe?.({ phase: "total", durationMs: performance.now() - started });
    self.postMessage({ id, result, ...(observe ? { diagnostics: { events, prefixCacheBytes: prefixCacheBytes() } } : {}) });
  } catch (error) {
    self.postMessage({
      id,
      error:
        error instanceof Error
          ? error.message
          : "Could not count tokens.",
    });
  } finally { controller?.abort(); runs.delete(id); }
};
