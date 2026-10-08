import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";

export function assertApprovedReport(decisions, moduleHash, suffix, bytes) {
  assert.equal(decisions.moduleHash, moduleHash, "Adoption decisions do not match the runtime");
  const digest = createHash("sha256").update(bytes).digest("hex");
  const rejection = decisions.rejectedReports[digest];
  assert(!rejection, `Rejected query plan: ${rejection?.reason}`);
  assert.equal(decisions.approvedReports[String(suffix)], digest,
    `Unreviewed query report for suffix${suffix}; record an adoption decision before publishing`);
  return digest;
}

/** Default measurements are isolated; only explicit calibration resumption uses the archive. */
export function measurementOutput(base, { resume = false, published = false } = {}) {
  assert(!(resume && published), "--published requires fresh measurements; do not combine with --resume");
  return resume ? base : new URL(`measurements/${Date.now()}-${randomUUID()}/`, base);
}

export async function readReusableMeasurement(file, expected, { resume = false, published = false } = {}) {
  assert(!(resume && published), "Performance measurements cannot reuse cached reports");
  if (!resume) return undefined;
  let old;
  try { old = JSON.parse(await readFile(file, "utf8")); }
  catch (error) { if (error.code === "ENOENT") return undefined; throw error; }
  if (old.complete && old.moduleHash === expected.moduleHash &&
      JSON.stringify(old.plan) === JSON.stringify(expected.plan) && old.inputHash === expected.inputHash &&
      JSON.stringify(old.options) === JSON.stringify(expected.options)) return old;
  return undefined;
}
