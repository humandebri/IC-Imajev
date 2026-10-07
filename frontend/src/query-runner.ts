import { concat, u32, sha, frame, unframe, front, checkCarry, C, H, P, CONV } from "./query-codec.ts";
import type { Header } from "./query-codec.ts";
import type { PrefixManifest, QueryClient } from "./inference-agent.ts";
import type { DecisionResult } from "./types.ts";

interface Measurement { state: Uint8Array; previous_hidden?: Uint8Array; conv?: Uint8Array; kv?: Uint8Array }
interface Dependencies {
  client: QueryClient; manifest: PrefixManifest; asset: (layer: number) => Promise<Uint8Array>;
  signal: AbortSignal; progress: (completed: number) => void;
  /** Test evidence hook, never retained by production UI. */
  observe?: (step: number, request: Uint8Array, reply: Uint8Array) => void;
}
function unwrap(value: unknown): Measurement {
  const reply = value as { Ok?: Measurement; Err?: string };
  if (reply.Err !== undefined) throw new Error(reply.Err);
  if (!reply.Ok) throw new Error("Invalid query result.");
  return reply.Ok;
}
export function validateInput(ids: number[], options: string[], manifest: PrefixManifest) {
  if (ids.length < 28 || ids.length > 84 || !ids.every(id => Number.isInteger(id) && id >= 0 && id < 248320) ||
      manifest.prefix.length !== 27 || !manifest.prefix.every((id, i) => ids[i] === id)) {
    throw new Error("Input must have the common prefix and 1–57 additional tokens (84 total maximum).");
  }
  if (options.length < 2 || options.length > 7 || options.some(o => !o.trim() || new TextEncoder().encode(o).length > 128) ||
      new Set(options.map(o => o.trim())).size !== options.length) throw new Error("Use 2–7 distinct options, each at most 128 UTF-8 bytes.");
}
export async function runQueryGraph(ids: number[], options: string[], d: Dependencies): Promise<DecisionResult> {
  validateInput(ids, options, d.manifest);
  const n = ids.length - P;
  const inputHash = await sha(new TextEncoder().encode(JSON.stringify(ids).replaceAll(",", ", ")));
  const header = (step: number, layer: number, op: string, encoding: string, dims: number[]): Header => ({
    version: 3, model: d.manifest.model, pack_hash: d.manifest.pack_hash, input_hash: inputHash,
    step, op, encoding, tensor: `model.language_model.layers.${layer}.post_attention_layernorm.weight`,
    aux: [`model.language_model.layers.${layer + 1}.input_layernorm.weight`], dims, scalars: [2, 1e-6],
  });
  async function send(step: number, method: string, h: Header, payload: Uint8Array, args: unknown[], expected: Header) {
    d.signal.throwIfAborted();
    const request = await frame(h, payload);
    d.signal.throwIfAborted();
    const reply = unwrap(await d.client.query(method, [request, ...args]));
    d.signal.throwIfAborted();
    const bytes = await unframe(reply.state, expected, true);
    d.observe?.(step, request, reply.state);
    return { reply, bytes };
  }
  const first = header(0, 0, "delta_mlp_stream_start_ids", "delta-mlp-start-exact-v1", [n, front(0), P]);
  const initial = await send(0, "step", first, concat(new Uint8Array([2]), ...ids.slice(P).map(u32), await d.asset(0)), [], { ...first, step: 1 });
  if (initial.bytes[0] !== 0 || initial.bytes.length <= 1 + 2 * CONV) throw new Error("Invalid initial reply.");
  let carry = initial.bytes.subarray(1, -2 * CONV);
  checkCarry(carry, n, front(0));
  d.progress(1);
  for (let layer = 0; layer <= 30; layer++) {
    const step = layer + 1, begin = front(layer), next = front(layer + 1);
    const attention = layer % 4 === 2;
    const terminal = layer === 30;
    const op = terminal ? "mlp_stream_complete_terminal" : attention ? "mlp_stream_complete_attention_full" : "mlp_stream_complete";
    const encoding = attention ? "mlp-attention-finish-exact-v1" : "mlp-stream-exact-v1";
    const h = header(step, layer, op, encoding, attention ? [n, P, begin] : [n, begin, H - begin]);
    checkCarry(carry, n, begin);
    const prefix = await d.asset(layer + 1);
    if (terminal) {
      const request = await frame(h, concat(new Uint8Array([3]), u32(carry.length), carry, prefix));
      d.signal.throwIfAborted();
      const value = await d.client.query("terminal_step_decision", [request, options]) as {
        Ok?: { measurement: Measurement; decision: Omit<DecisionResult, "value"> & { value: string[] } }; Err?: string;
      };
      d.signal.throwIfAborted();
      if (!value.Ok) throw new Error(value.Err ?? "Invalid terminal response.");
      const payload = await unframe(value.Ok.measurement.state, { ...h, step: 32 }, true);
      if (payload[0] !== 0 || payload.length !== 1 + 2 * (2 * C + n * 2048)) throw new Error("Invalid terminal state.");
      d.observe?.(step, request, value.Ok.measurement.state);
      const result = value.Ok.decision;
      const probabilities = [...result.probabilities, result.unknown_probability];
      const selected = result.value[0] ?? null;
      if (result.probabilities.length !== options.length || result.value.length > 1 ||
          probabilities.some(p => !Number.isFinite(p) || p < 0 || p > 1) ||
          Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) > 0.0001 ||
          (selected !== null && !options.includes(selected)) || result.abstained !== (selected === null)) {
        throw new Error("Invalid decision response.");
      }
      d.progress(32);
      return { value: selected, probabilities: result.probabilities,
        unknown_probability: result.unknown_probability, abstained: result.abstained };
    }
    const expected = header(step + 1, layer + 1, "mlp_stream_prepare", "mlp-stream-exact-v1", [n, 0, next]);
    const payload = attention ? concat(new Uint8Array([3]), u32(carry.length), carry, prefix) : carry;
    const { reply, bytes } = await send(step, attention ? "attention_mlp_front" : "mlp_delta_front", h, payload,
      attention ? [next] : [prefix, P, next], expected);
    if (reply.previous_hidden?.length !== 2 * n * C ||
        (attention ? reply.kv?.length !== 2 * n * 2048 : reply.conv?.length !== 2 * CONV)) throw new Error("Invalid bridge output shape.");
    checkCarry(bytes, n, next);
    carry = bytes;
    d.progress(step + 1);
  }
  throw new Error("Inference ended without a decision.");
}
