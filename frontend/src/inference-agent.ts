import { HttpAgent, StatePaths } from "@icp-sdk/core/agent";
import { Principal } from "@icp-sdk/core/principal";
import { IDL } from "@icp-sdk/core/candid";
import release from "./inference-release.json" with { type: "json" };
import { sha } from "./query-codec.ts";

const blob = IDL.Vec(IDL.Nat8);
const measurementFields = { state: blob, instructions: IDL.Nat64,
  stable_read_bytes: IDL.Nat64, heap_pages: IDL.Nat64, stable_pages: IDL.Nat64 };
const measurement = IDL.Record(measurementFields);
const decision = IDL.Record({ value: IDL.Opt(IDL.Text), probabilities: IDL.Vec(IDL.Float32),
  unknown_probability: IDL.Float32, abstained: IDL.Bool, raw_logits: IDL.Vec(IDL.Float32),
  instructions: IDL.Nat64, calibration_version: IDL.Text });
const result = (type: IDL.Type) => IDL.Variant({ Ok: type, Err: IDL.Text });
const bridge = (field: string) => IDL.Record({ ...measurementFields, previous_hidden: blob,
  [field]: blob, spans: IDL.Vec(IDL.Tuple(IDL.Text, IDL.Nat64)) });
export const methods: Record<string, { args: IDL.Type[]; reply: IDL.Type[] }> = {
  step: { args: [blob], reply: [result(measurement)] },
  mlp_delta_front: { args: [blob, blob, IDL.Nat32, IDL.Nat32], reply: [result(bridge("conv"))] },
  attention_mlp_front: { args: [blob, IDL.Nat32], reply: [result(bridge("kv"))] },
  terminal_step_decision: { args: [blob, IDL.Vec(IDL.Text)], reply: [result(IDL.Record({ measurement, decision }))] },
  pack_status: { args: [], reply: [IDL.Record({ model: IDL.Text, pack_hash: IDL.Text,
    bytes: IDL.Nat64, received: IDL.Nat64, hashed: IDL.Nat64, ready: IDL.Bool,
    chunks: IDL.Vec(IDL.Nat64) })] },
  weight_cache_status: { args: [], reply: [IDL.Record({ bytes: IDL.Nat64, names: IDL.Vec(IDL.Text),
    preparation_instructions: IDL.Nat64, rope_bytes: IDL.Opt(IDL.Nat64),
    activation_bytes: IDL.Opt(IDL.Nat64), paired_weight_bytes: IDL.Opt(IDL.Nat64) })] },
};
export interface QueryClient {
  /** Decoded replies after the IC agent has verified the node signature. */
  query: (method: string, args: unknown[]) => Promise<unknown>;
  checkModule: () => Promise<void>;
}
export async function createQueryClient(signal: AbortSignal): Promise<QueryClient> {
  const timedFetch: typeof fetch = async (input, init) => {
    const combined = AbortSignal.any([signal, AbortSignal.timeout(30_000)]);
    const response = await fetch(input, { ...init, signal: combined });
    if (!response.ok && [429, 502, 503, 504].includes(response.status)) {
      const error = new Error(`IC gateway busy (${response.status}). Please try again.`);
      throw error;
    }
    return response;
  };
  const agent = await HttpAgent.create({ host: release.host, fetch: timedFetch,
    retryTimes: 0, verifyQuerySignatures: true });
  const canister = Principal.fromText(release.canister);
  return {
    async query(method, args) {
      signal.throwIfAborted();
      const schema = methods[method];
      if (!schema) throw new Error("Unsupported query method.");
      const bytes = IDL.encode(schema.args, args);
      if (bytes.byteLength >= 1_990_000) throw new Error("Query request too large.");
      const response = await agent.query(canister, { methodName: method, arg: bytes });
      signal.throwIfAborted();
      if (response.status !== "replied") throw new Error("IC query rejected. Please try again.");
      if (response.reply.arg.byteLength >= 1_990_000) throw new Error("Query reply too large.");
      return IDL.decode(schema.reply, response.reply.arg)[0];
    },
    async checkModule() {
      const path = StatePaths.canisterModuleHash(canister);
      const response = await agent.readState({ canisterId: canister }, { paths: [path] });
      const hash = response.values.get(path);
      if (!hash || Array.from(hash, b => b.toString(16).padStart(2, "0")).join("") !== release.module_hash) {
        throw new Error("Inference runtime changed. Reload after deployment completes.");
      }
    },
  };
}

export interface PrefixManifest {
  model: string; pack_hash: string; model_bytes: number; prefix: number[];
  assets: Record<string, { file: string; bytes: number; sha256: string }>;
}
export async function loadPrefix(base: string, signal: AbortSignal) {
  const response = await fetch(`${base}manifest.json`, { signal });
  if (!response.ok) throw new Error("Could not load prefix settings.");
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (await sha(bytes) !== release.manifest_sha256) throw new Error("Prefix manifest mismatch.");
  const manifest = JSON.parse(new TextDecoder().decode(bytes)) as PrefixManifest;
  return { manifest, async asset(layer: number) {
    signal.throwIfAborted();
    const entry = manifest.assets[String(layer)];
    const response = await fetch(`${base}${entry.file}`, { signal });
    if (!response.ok) throw new Error("Could not load prefix state.");
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== entry.bytes || await sha(bytes) !== entry.sha256) throw new Error("Prefix state mismatch.");
    return bytes;
  } };
}
