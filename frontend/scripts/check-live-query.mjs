/** Explicit, read-only mainnet browser test. Not part of the default test suite. */
import { chromium } from "@playwright/test";
import { readFile, mkdir, writeFile } from "node:fs/promises";
const root = new URL("../../", import.meta.url);
const records = JSON.parse(await readFile(new URL("artifacts/text-short-v2/inputs.json", root), "utf8"));
const report = JSON.parse(await readFile(new URL("artifacts/mainnet-prefix27-upgrade-20261007/anonymous-query-653/report.json", root), "utf8"));
const browser = await chromium.launch({ headless: true });
const output = new URL("artifacts/browser-query-test-20261007/", root);
await mkdir(output, { recursive: true });
const fixture = records.records[2];
const first = { state: "This proposal mints 250,000,000 tokens to one account.",
  question: "Could this token mint concentrate token control?", options: fixture.options };
const second = { state: "The limit changes from 5 to 10.", question: "Does the limit increase?", options: ["yes", "no"] };
try {
  const results = await Promise.all([first, second].map(async (input, index) => {
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
    const selected = await page.locator(".actual-result h3").innerText();
    const probabilities = await page.locator("meter").evaluateAll(nodes => nodes.map(n => n.value));
    if (writes.length || errors.length) throw new Error(`Unexpected writes/errors: ${JSON.stringify({ writes, errors })}`);
    if (queries !== 34) throw new Error(`Expected 32 inference + 2 readiness queries, got ${queries}`);
    if (index === 0 && (selected !== report.decision.value || JSON.stringify(probabilities) !== JSON.stringify([...report.decision.probabilities, report.decision.unknown_probability]))) {
      throw new Error("Browser result differs from verified mainnet evidence.");
    }
    await page.screenshot({ path: new URL(`run-${index}.png`, output).pathname, fullPage: true });
    const result = { input, selected, probabilities, queries, writes, errors, seconds: (Date.now() - started) / 1000 };
    console.log(JSON.stringify(result));
    await context.close();
    return result;
  }));
  await writeFile(new URL("report.json", output), JSON.stringify({ complete: true, independentConcurrentBrowsers: 2, results }, null, 2));
} finally { await browser.close(); }
