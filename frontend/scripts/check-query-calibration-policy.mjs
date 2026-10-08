import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import decisions from "./query-plan-decisions.json" with { type: "json" };
import { assertApprovedReport, measurementOutput, readReusableMeasurement } from "./query-calibration-policy.mjs";

const cwd = new URL("../", import.meta.url);
const cli = (...args) => spawnSync(process.execPath, ["--experimental-strip-types", ...args], {cwd, encoding:"utf8"});
const catalogURL = new URL("../src/query-profiles.json", import.meta.url);
const before = await readFile(catalogURL);
const accepted = cli("scripts/prepare-query-profiles.mjs", "--balanced", "--through=68", "--check");
assert.equal(accepted.status, 0, accepted.stderr);
const rejected = cli("scripts/prepare-query-profiles.mjs", "--balanced");
assert.notEqual(rejected.status, 0);
assert.match(rejected.stderr, /Rejected query plan:.*96-token61-query/);
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
console.log("Rejected/unreviewed plans cannot be published; fresh comparisons ignore caches; explicit resumption checks all bindings.");
