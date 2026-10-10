import assert from "node:assert/strict";

/** Offline candidate search. Predictions never authorize a production plan. */
export function optimizeQueryPlan(baseline, split, probeBase, probeSplit, makePlan) {
  const n = baseline.n, target = 3_850_000_000;
  for (const report of [baseline, split, probeBase, probeSplit]) {
    assert(report.complete && report.moduleHash === baseline.moduleHash);
  }
  assert.equal(baseline.inputHash, split.inputHash);
  assert.equal(baseline.n, split.n);
  assert.equal(probeBase.inputHash, probeSplit.inputHash);
  assert.equal(probeBase.n, probeSplit.n);
  assert.equal(split.count, 63);
  const points = Array.from({length: 35}, (_, i) => (i + 1) * 256);
  const sizeSlope = n * (1 + 4 / 256);
  const bytes = q => q.requestBytes + q.replyBytes;
  const better = (a, b) => !b || a.count < b.count || (a.count === b.count && a.bytes < b.bytes);
  const unit0 = (probeBase.queries[0].instructions - probeSplit.queries[0].instructions) /
    (probeBase.plan.fronts[0] - probeSplit.plan.fronts[0]) * n / probeBase.n;
  assert(unit0 > 0);
  let states = new Map();
  for (const front of points) {
    if (baseline.queries[0].instructions + unit0 * (front - baseline.plan.fronts[0]) > target) continue;
    states.set(front, {count: 1, bytes: bytes(baseline.queries[0]) + sizeSlope * (front - baseline.plan.fronts[0]), fronts: [front], completions: []});
  }
  for (let layer = 0; layer <= 30; layer++) {
    const terminal = layer === 30;
    const a = baseline.queries[baseline.count === 32 ? layer + 1 : 2 * layer + 2];
    const b = split.queries[2 * layer + 2], chunk = split.queries[2 * layer + 1];
    const originalDone = baseline.plan.completions[layer];
    const originalNext = terminal ? 0 : baseline.plan.fronts[layer + 1];
    const splitNext = terminal ? 0 : split.plan.fronts[layer + 1];
    const unit = (a.instructions - b.instructions) /
      (split.plan.completions[layer] - originalDone + originalNext - splitNext);
    assert(unit > 0 && Number.isFinite(unit));
    const chunkRange = split.plan.completions[layer] - split.plan.fronts[layer];
    const nextStates = new Map();
    for (const [front, state] of states) for (const done of points) {
      if (done < front) continue;
      const extra = done > front;
      // Retain the measured overhead even for smaller chunks.
      if (extra && chunk.instructions + unit * Math.max(0, done - front - chunkRange) > target) continue;
      for (const next of terminal ? [0] : points) {
        if (a.instructions + unit * (originalDone - done + next - originalNext) > target) continue;
        const chunkBytes = extra ? bytes(chunk) + sizeSlope *
          (front - split.plan.fronts[layer] + done - split.plan.completions[layer]) : 0;
        const bridgeBytes = bytes(a) + sizeSlope * (done - originalDone + next - originalNext);
        const candidate = {count: state.count + 1 + Number(extra), bytes: state.bytes + chunkBytes + bridgeBytes,
          fronts: [...state.fronts, terminal ? 256 : next], completions: [...state.completions, done]};
        if (better(candidate, nextStates.get(next))) nextStates.set(next, candidate);
      }
    }
    assert(nextStates.size > 0, `No safe predicted schedule at layer ${layer}`);
    states = nextStates;
  }
  const result = states.get(0);
  return {...makePlan(n), fronts: result.fronts, completions: result.completions};
}

/** Predict a multi-chunk schedule from a fresh conservative run; remeasure before adoption. */
export function optimizeStreamQueryPlan(reference, makePlan) {
  assert(reference.complete);
  const n = reference.n, plan = makePlan(n);
  assert(n > 45 && n <= 91);
  const chunks = reference.queries.filter(q => q.op === "mlp_stream_next");
  assert(chunks.length > 0);
  const unit = Math.max(...chunks.map(q => q.instructions / q.dims[2])) * 1.03;
  const target = 3_750_000_000;
  const floor = x => Math.floor(x / 256) * 256;
  const ceil = x => Math.ceil(x / 256) * 256;
  plan.fronts[0] = Math.max(256, Math.min(8960, floor(256 + (target - reference.queries[0].instructions) / unit)));
  for (let layer = 0; layer <= 30; layer++) {
    const q = reference.queries.find(q => q.layer === layer && q.op.startsWith("mlp_stream_complete"));
    assert(q, `Missing bridge measurement at layer ${layer}`);
    const done = Math.max(plan.fronts[layer], Math.min(8960, ceil(8960 - (target - q.instructions) / unit)));
    plan.completions[layer] = done;
    if (layer < 30) {
      const attention = layer % 4 === 2;
      const replyCap = floor(((1_970_000 - (attention ? 0 : 49_152)) / n - (attention ? 17_704 : 13_608)) / 1.015625);
      plan.fronts[layer + 1] = Math.max(256, Math.min(8960, replyCap,
        floor(256 + done - 8960 + (target - q.instructions) / unit)));
    }
  }
  return plan;
}
