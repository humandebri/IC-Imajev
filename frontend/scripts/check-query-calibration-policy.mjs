import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import decisions from "./query-plan-decisions.json" with { type: "json" };
import { assertApprovedReport, assertApprovedProfile, profileDigest, measurementOutput, readReusableMeasurement } from "./query-calibration-policy.mjs";
import { queryCount, validatePlan } from "../src/query-plan.ts";

const cwd = new URL("../", import.meta.url);
const cli = (...args) => spawnSync(process.execPath, ["--experimental-strip-types", ...args], {cwd, encoding:"utf8"});
const catalogURL = new URL("../src/query-profiles.json", import.meta.url);
const before = await readFile(catalogURL);
const catalog = JSON.parse(before);
const suffix = 91, plan = catalog.plans[suffix], evidence = catalog.evidence[suffix];
assertApprovedProfile(decisions, catalog.moduleHash, suffix, plan, evidence);
// Retain the approved report hash and query count while changing only the schedule.
for (const edit of [p => p.fronts[0] += 256, p => p.completions[0] -= 256]) {
  const changed = structuredClone(plan); edit(changed);
  const full = { ...changed, moduleHash: catalog.moduleHash, suffix };
  validatePlan(full, suffix);
  assert.equal(queryCount(full), evidence.queries);
  assert.throws(() => assertApprovedProfile(decisions, catalog.moduleHash, suffix, changed, evidence), /differs from the approved profile/);
}
for (const edit of [e => e.maxInstructions--, e => e.maxRequestBytes--,
  e => e.maxReplyBytes--, e => e.finalPayloadHash = "0".repeat(64),
  e => e.reference = "unchanged-baseline"]) {
  const changed = structuredClone(evidence); edit(changed);
  assert.throws(() => assertApprovedProfile(decisions, catalog.moduleHash, suffix, plan, changed), /differs from the approved profile/);
}
assert.throws(() => assertApprovedProfile({ ...decisions, approvedProfiles: {} }, catalog.moduleHash, suffix, plan, evidence), /Missing approved profile/);
assert.equal(profileDigest(catalog.moduleHash, suffix, plan, evidence),
  profileDigest(catalog.moduleHash, suffix, { completions: plan.completions, fronts: plan.fronts },
    Object.fromEntries(Object.entries(evidence).reverse())), "Object property order must not change approval");
const accepted = cli("scripts/prepare-query-profiles.mjs", "--check");
assert.equal(accepted.status, 0, accepted.stderr);
const rejectedBytes = Buffer.from("rejected measurement");
const rejectedDigest = (await import("node:crypto")).createHash("sha256").update(rejectedBytes).digest("hex");
assert.throws(() => assertApprovedReport({...decisions, rejectedReports: {[rejectedDigest]: {reason: "query budget exceeded"}}}, decisions.moduleHash, suffix, rejectedBytes), /Rejected query plan:.*query budget exceeded/);
assert.deepEqual(await readFile(catalogURL), before, "A rejected candidate must never modify the catalog");
assert.throws(() => assertApprovedReport(decisions, decisions.moduleHash, 58, Buffer.from("unreviewed measurement")), /Unreviewed query report/);
assert.throws(() => assertApprovedReport(decisions, "other-module", 58, Buffer.from("x")), /runtime/);

const invalidResume = cli("scripts/profile-query-plans.mjs", "--published", "--resume", "69");
assert.notEqual(invalidResume.status, 0);
assert.match(invalidResume.stderr, /--published requires fresh measurements/);

const dir = await mkdtemp(join(tmpdir(), "imajev-calibration-policy-"));
try {
  const file = join(dir,"report.json");
  const expected = {moduleHash:"pinned",plan:{fronts:[256]},inputHash:"input",options:["yes","no"]};
  const report = {complete:true,...expected,seconds:12345};
  await writeFile(file,JSON.stringify(report));
  assert.equal(await readReusableMeasurement(file,expected),undefined,"Default mode must not return stale timings");
  assert.equal(await readReusableMeasurement(file,expected,{published:true}),undefined);
  assert.deepEqual(await readReusableMeasurement(file,expected,{resume:true}),report);
  for(const change of [{moduleHash:"wrong"},{inputHash:"other"},{plan:{fronts:[512]}},{options:["no","yes"]}]) {
    assert.equal(await readReusableMeasurement(file,{...expected,...change},{resume:true}),undefined);
  }
  await writeFile(file,"invalid old cache");
  assert.equal(await readReusableMeasurement(file,expected,{published:true}),undefined,"Performance mode must not even parse old reports");
  await assert.rejects(readReusableMeasurement(file,expected,{published:true,resume:true}), /cannot reuse/);
  const base=new URL(`file://${dir}/`);
  const first=measurementOutput(base),second=measurementOutput(base);
  assert.notEqual(first.href,second.href,"Repeated measurements must have distinct evidence directories");
  assert(first.href.startsWith(new URL("measurements/",base).href));
  assert.equal(measurementOutput(base,{resume:true}).href,base.href);
  assert.throws(()=>measurementOutput(base,{published:true,resume:true}),/fresh measurements/);
} finally { await rm(dir,{recursive:true,force:true}); }
console.log("Published schedules and evidence match approved profiles; altered plans with unchanged report hashes/counts are rejected; fresh comparisons ignore caches.");
