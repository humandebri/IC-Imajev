import assert from "node:assert/strict";
import profiles from "../src/query-profiles.json" with { type: "json" };
import release from "../src/inference-release.json" with { type: "json" };
import { selectPlan, queryCount } from "../src/query-plan.ts";
import decisions from "./query-plan-decisions.json" with { type: "json" };
assert.equal(profiles.moduleHash,release.module_hash);
assert.equal(decisions.moduleHash,release.module_hash);
assert(Number.isInteger(profiles.maxSuffix)&&profiles.maxSuffix>=57&&profiles.maxSuffix<=69);
assert.equal(profiles.targetInstructions,4_000_000_000);
assert.equal(Object.keys(profiles.plans).length,profiles.maxSuffix);
assert.equal(Object.keys(profiles.evidence).length,profiles.maxSuffix);
for(let n=1;n<=profiles.maxSuffix;n++) {
  const plan=selectPlan(n),e=profiles.evidence[String(n)];
  assert.equal(e.reportSHA256,decisions.approvedReports[String(n)],`Unreviewed published plan at suffix${n}`);
  assert(!decisions.rejectedReports[e.reportSHA256],`Rejected published plan at suffix${n}`);
  assert.equal(queryCount(plan),e.queries);
  assert(e.maxInstructions>0&&e.maxInstructions<=profiles.targetInstructions);
  assert(e.maxRequestBytes>0&&e.maxRequestBytes<1_990_000&&e.maxReplyBytes>0&&e.maxReplyBytes<1_990_000);
  assert(/^[0-9a-f]{64}$/.test(e.reportSHA256)&&/^[0-9a-f]{64}$/.test(e.finalPayloadHash));
  assert(["unchanged-baseline","separate-query-schedule"].includes(e.reference));
}
console.log(`Validated all ${profiles.maxSuffix} measured query plans against the runtime pin and budgets.`);
