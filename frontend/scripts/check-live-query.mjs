/** Explicit, read-only mainnet browser test. Not part of the default test suite. */
import { queryCount, selectPlan } from "../src/query-plan.ts";
import release from "../src/inference-release.json" with { type: "json" };
import { chromium } from "@playwright/test";
import { readFile, mkdir, writeFile } from "node:fs/promises";
const root = new URL("../../", import.meta.url);
const evidenceRoot = new URL("artifacts/prefix5-local-20261009/", root);
const examples = JSON.parse(await readFile(new URL("../src/boom-examples.generated.json", import.meta.url), "utf8"));
const inputs = examples.map(e => e.input);
const expected = await Promise.all(examples.map(async e => {
  const proofUrl = e.id === "boom-617"
    ? new URL("artifacts/boom617-malicious-validation-20261009/measurements/final-question.json", root)
    : new URL(`measurements/${e.id}.json`, evidenceRoot);
  const proof = JSON.parse(await readFile(proofUrl, "utf8"));
  if (!proof.complete || proof.moduleHash !== release.module_hash) throw new Error("Local reference runtime mismatch");
  return proof.result;
}));
const boundary = JSON.parse(await readFile(new URL("extra-inputs.json", evidenceRoot), "utf8")).records[3];
inputs.push(boundary.source_input);
expected.push(JSON.parse(await readFile(new URL("full-extra-3/report.json", evidenceRoot), "utf8")).decision_query.ok.decision);
const browser = await chromium.launch({ headless: true });
const output = new URL(process.env.QUERY_TEST_OUTPUT ?? "artifacts/prefix5-mainnet-browser/", root);
await mkdir(output, { recursive: true });
try {
  const results = await Promise.all(inputs.map(async (input, index) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    let queries = 0;
    const writes = [], errors = [];
    page.on("request", r => {
      if (/\/query$/.test(r.url())) queries++;
      if (/\/call$|\/call\//.test(r.url())) writes.push(r.url());
    });
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(process.env.PLAYGROUND_URL ?? "http://127.0.0.1:5173");
    await page.getByLabel("Context").fill(input.state);
    await page.getByLabel("Question").fill(input.question);
    while (await page.locator(".option-control input").count() < input.options.length) await page.getByRole("button", { name: "Add option" }).click();
    for (let i = 0; i < input.options.length; i++) await page.locator(".option-control input").nth(i).fill(input.options[i]);
    await page.getByRole("button", { name: "Run inference" }).waitFor();
    await page.waitForFunction(() => !document.querySelector(".primary-button").disabled, null, { timeout: 30_000 });
    const started = Date.now();
    await page.getByRole("button", { name: "Run inference" }).click();
    await page.waitForFunction(() => document.querySelector('.result-panel[data-state="done"]') || document.querySelector('[role="alert"]'), null, { timeout: 360_000 });
    const error = await page.locator('[role="alert"]').allTextContents();
    if (error.length) throw new Error(`Browser run ${index}: ${error.join(" ")}`);
    const heading = await page.locator(".actual-result h3").innerText();
    const selected = await page.locator(".result-label").innerText() === "Abstained" ? null
      : heading.replace(/\s*·\s*[\d,.]+%$/, "");
    const probabilities = await page.locator("meter").evaluateAll(nodes => nodes.map(n => n.value));
    if (writes.length || errors.length) throw new Error(`Unexpected writes/errors: ${JSON.stringify({ writes, errors })}`);
    const totalText = await page.locator(".token-counts dd").innerText();
    const n = Number(totalText) - release.prefix_tokens;
    const count = queryCount(selectPlan(n));
    if (queries !== count + 2) throw new Error(`Expected ${count} inference + 2 readiness queries, got ${queries}`);
    const reference = expected[index];
    if (selected !== reference.value || JSON.stringify(probabilities) !== JSON.stringify([...reference.probabilities, reference.unknown_probability])) {
      throw new Error("Browser result differs from the verified five-token/full-processing reference.");
    }
    await page.screenshot({ path: new URL(`run-${index}.png`, output).pathname, fullPage: true });
    const result = { input, selected, probabilities, queries, writes, errors, seconds: (Date.now() - started) / 1000 };
    console.log(JSON.stringify(result));
    await context.close();
    return result;
  }));
  await writeFile(new URL("report.json", output), JSON.stringify({ complete: true, independentConcurrentBrowsers: results.length, results }, null, 2));
} finally { await browser.close(); }
