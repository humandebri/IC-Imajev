/** Explicit read-only mainnet benchmark: two inputs, cold/warm, three paired rounds (24 runs). */
import { chromium } from "@playwright/test";
import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";

const root = new URL("../../", import.meta.url);
const json = async path => JSON.parse(await readFile(new URL(path, root), "utf8"));
const profiles = await json("frontend/src/query-profiles.json");
const fixtures = [
  { name: "short", input: { state: "The limit changes from 5 to 10.", question: "Does the limit increase?", options: ["yes", "no"] },
    expected: (await json("artifacts/browser-query-test-20261007/report.json")).results[1] },
  { name: "96-token", input: await json("artifacts/adaptive-query-20261008/real-input-n69.json"),
    expected: (await json("artifacts/adaptive-query-20261008/real-n69-verified.json")).reference.result },
];
const output = new URL(process.env.PREFETCH_OUTPUT ?? "artifacts/frontend-prefetch-20261008/", root);
await mkdir(output, { recursive: true });
const variants = [
  { label: "baseline", url: process.env.BASELINE_URL ?? "http://127.0.0.1:4191", dist: ".cache/prefetch-baseline-dist/" },
  { label: "candidate", url: process.env.CANDIDATE_URL ?? "http://127.0.0.1:4192", dist: "frontend/dist/" },
];
for (const variant of variants) {
  const names = await readdir(new URL(`${variant.dist}assets/`, root));
  variant.worker = names.find(name => /^tokenizer\.worker-.*\.js$/.test(name));
  assert(variant.worker, "Missing production worker");
  variant.sha256 = createHash("sha256").update(await readFile(new URL(`${variant.dist}assets/${variant.worker}`, root))).digest("hex");
}
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const results = [];
const browser = await chromium.launch({ headless: true });
const summarize = () => fixtures.flatMap(fixture => ["cold", "warm"].map(temperature => {
  const times = label => results.filter(r => r.fixture === fixture.name && r.temperature === temperature && r.variant === label).map(r => r.ms);
  const before = times("baseline"), after = times("candidate");
  return { fixture: fixture.name, temperature, baselineRuns: before.length, candidateRuns: after.length,
    baselineMedianMs: before.length ? median(before) : null, candidateMedianMs: after.length ? median(after) : null,
    improvementPercent: before.length && after.length ? 100 * (1 - median(after) / median(before)) : null };
}));
const save = async (complete, failure) => writeFile(new URL("report.json", output), JSON.stringify({
  complete, variants, results, summary: summarize(), failure,
  note: "Signed mainnet queries only. Query durations include encoding/signature verification. Phase times overlap. No model/update changes.",
}, null, 2) + "\n");
try {
  for (let round = 0; round < 3; round++) {
    for (const fixture of fixtures) {
      // Reverse order on alternate rounds to reduce order bias. Never run pairs concurrently.
      for (const variant of round % 2 ? [...variants].reverse() : variants) {
        const context = await browser.newContext();
        const writes = [], errors = [];
        let queryRequests = 0;
        context.on("request", request => {
          if (/\/query$/.test(request.url())) queryRequests++;
          if (/\/call(?:\/|$)/.test(request.url())) writes.push(request.url());
        });
        const page = await context.newPage();
        page.on("pageerror", error => errors.push(error.message));
        const cdp = await context.newCDPSession(page);
        await cdp.send("Network.clearBrowserCache");
        await page.route("**/__prefetch-benchmark", route => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>Inference timing harness</title>" }));
        await page.goto(`${variant.url}/__prefetch-benchmark`);
        try {
          for (const temperature of ["cold", "warm"]) {
            const beforeRequests = queryRequests;
            const run = await page.evaluate(async ({ worker, input, temperature }) => {
              if (temperature === "cold") window.perfWorker = new Worker(worker, { type: "module" });
              const send = message => new Promise((resolve, reject) => {
                const timer = setTimeout(() => { cleanup(); reject(new Error("Benchmark timeout")); }, 360000);
                const listener = event => {
                  if (event.data.id !== message.id || event.data.completed !== undefined) return;
                  cleanup();
                  if (event.data.error) reject(new Error(event.data.error)); else resolve(event.data);
                };
                const failed = event => { cleanup(); reject(new Error(event.message ?? "Worker error")); };
                const cleanup = () => { clearTimeout(timer); window.perfWorker.removeEventListener("message", listener); window.perfWorker.removeEventListener("error", failed); };
                window.perfWorker.addEventListener("message", listener);
                window.perfWorker.addEventListener("error", failed);
                window.perfWorker.postMessage(message);
              });
              const readyStart = performance.now();
              const count = await send({ id: 1, op: "count", input });
              const readyMs = performance.now() - readyStart;
              const start = performance.now();
              const response = await send({ id: 2, op: "run", input, diagnostics: true });
              return { ...response, counts: count.counts, readyMs, ms: performance.now() - start };
            }, { worker: `${variant.url}/assets/${variant.worker}`, input: fixture.input, temperature });
            // Historical browser evidence includes abstention as the last meter value.
            const expectedProbabilities = fixture.expected.probabilities.slice(0, fixture.input.options.length);
            assert.deepEqual(run.result.probabilities, expectedProbabilities);
            assert.equal(run.result.value, fixture.expected.value ?? fixture.expected.selected);
            assert.equal(run.result.unknown_probability, fixture.expected.unknown_probability ?? fixture.expected.probabilities.at(-1));
            const plan = profiles.plans[String(run.counts.total - 27)];
            const expectedQueries = 32 + plan.completions.filter((done, layer) => done > plan.fronts[layer]).length + 2;
            assert.equal(queryRequests - beforeRequests, expectedQueries);
            assert.deepEqual(writes, []); assert.deepEqual(errors, []);
            if (fixture.name === "96-token") assert.equal(run.counts.total, 96);
            const events = run.diagnostics.events;
            const assetFetches = events.filter(e => e.phase === "prefix-fetch" && e.layer !== undefined).length;
            if (variant.label === "candidate" && temperature === "warm") {
              assert.equal(events.filter(e => e.phase === "prefix-fetch").length, 0, "Warm prefix fetches");
              assert.equal(events.filter(e => e.phase === "prefix-hash").length, 0, "Warm prefix hashes");
            }
            results.push({ variant: variant.label, fixture: fixture.name, round, temperature, ...run,
              queries: queryRequests - beforeRequests, assetFetches });
            await save(false);
            console.log(JSON.stringify({ variant: variant.label, fixture: fixture.name, round, temperature,
              seconds: run.ms / 1000, assetFetches, prefixWaitMs: events.filter(e => e.phase === "prefix-wait").reduce((sum, e) => sum + e.durationMs, 0) }));
          }
        } finally { await context.close(); }
      }
    }
  }
  await save(true);
  console.log(JSON.stringify({ complete: true, summary: summarize() }));
} catch (error) {
  await save(false, String(error));
  throw error;
} finally { await browser.close(); }
