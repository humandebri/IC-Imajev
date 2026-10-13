import release from "./inference-release.json" with { type: "json" };
import profiles from "./query-profiles.json" with { type: "json" };
import { front, H, P } from "./query-codec.ts";

/** Only opaque MLP generation is split; no model arithmetic runs on the host. */
export interface QueryPlan {
  moduleHash: string;
  suffix: number;
  fronts: number[];
  completions: number[];
}
export const MAX_SUFFIX = 91;
// One aligned opaque chunk covers the maximum legal gap (8960 - 256 rows).
export const MLP_CHUNK_ROWS = 8704;
export const MAX_TOKENS = P + MAX_SUFFIX;
export function baselinePlan(suffix: number): QueryPlan {
  // Conservative schedule used to measure this release before selecting plans.
  const split = suffix > 45;
  return { moduleHash: release.module_hash, suffix,
    fronts: Array.from({ length: 32 }, (_, layer) => split ? 256 : front(layer)),
    completions: Array.from({ length: 31 }, (_, layer) => split ? 8960 : front(layer)) };
}
export function validatePlan(plan: QueryPlan, suffix: number) {
  const aligned = (n: number) => Number.isInteger(n) && n > 0 && n < H && n % 256 === 0;
  if (plan.moduleHash !== release.module_hash || plan.suffix !== suffix ||
      !Number.isInteger(suffix) || suffix < 1 || suffix > 91 ||
      plan.fronts.length !== 32 || plan.completions.length !== 31 ||
      !plan.fronts.every(aligned) || !plan.completions.every((n, layer) => aligned(n) && n >= plan.fronts[layer])) {
    throw new Error("Invalid query execution plan or runtime mismatch.");
  }
}
export function queryCount(plan: QueryPlan) {
  validatePlan(plan, plan.suffix);
  return 32 + plan.completions.reduce((sum, n, layer) => sum + Math.ceil((n - plan.fronts[layer]) / MLP_CHUNK_ROWS), 0);
}
export function selectPlan(suffix: number, moduleHash = release.module_hash): QueryPlan {
  if (moduleHash !== release.module_hash || profiles.moduleHash !== moduleHash) throw new Error("Query plans do not match the inference runtime.");
  if (!Number.isInteger(suffix) || suffix < 1 || suffix > MAX_SUFFIX) throw new Error(`Input exceeds ${MAX_TOKENS} tokens.`);
  const entry = (profiles.plans as Record<string, { fronts: number[]; completions: number[] }>)[String(suffix)];
  if (!entry) throw new Error("No verified query plan for this token count.");
  const plan = { moduleHash, suffix, ...entry };
  validatePlan(plan, suffix);
  return { ...plan, fronts: [...plan.fronts], completions: [...plan.completions] };
}
