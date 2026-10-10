// Read-only model evaluation. A caller may use a built legacy-prefix preview.
import { chromium } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
const examples = JSON.parse(await readFile(new URL('../src/boom-examples.generated.json', import.meta.url)));
const sample = examples.find(example => example.id === 'boom-617');
const directory = new URL(process.env.BOOM617_TEST_OUTPUT ?? '../../artifacts/boom-617-malicious-browser-20261009/', import.meta.url);
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  const writes = [], errors = [];
  page.on('request', request => { if (/\/call(?:\/|$)/.test(request.url())) writes.push(request.url()); });
  page.on('pageerror', error => errors.push(error.message));
  const url = process.env.PLAYGROUND_URL ?? 'http://127.0.0.1:5173';
  await page.goto(url);
  await page.getByLabel('Context').fill(sample.input.state);
  await page.getByLabel('Question').fill(sample.input.question);
  while (await page.locator('.option-control input').count() < sample.input.options.length) {
    await page.getByRole('button', { name: 'Add option' }).click();
  }
  for (let index = 0; index < sample.input.options.length; index++) {
    await page.locator('.option-control input').nth(index).fill(sample.input.options[index]);
  }
  await page.waitForFunction(() => !document.querySelector('.primary-button').disabled, null, { timeout: 30000 });
  const tokens = await page.locator('.token-counts dd').innerText();
  await page.getByText('Count details', { exact: true }).click();
  const countDetails = await page.locator('.token-details').innerText();
  await page.getByRole('button', { name: 'Run inference' }).click();
  await page.waitForFunction(() => document.querySelector('.result-panel[data-state="done"]') || document.querySelector('[role="alert"]'), null, { timeout: 360000 });
  const alerts = await page.locator('[role="alert"]').allTextContents();
  const selected = alerts.length ? null : (await page.locator('.actual-result h3').innerText()).replace(/\s*·.*$/, '');
  const report = { url, input: sample.input, tokens, countDetails, alerts, writes, errors, selected,
    result: await page.locator('.result-panel').innerText(), evidence: sample.audit.participation_snapshot };
  await writeFile(new URL('report.json', directory), JSON.stringify(report, null, 2) + '\n');
  await page.screenshot({ path: new URL('result.png', directory).pathname, fullPage: true });
  console.log(JSON.stringify(report));
  if (alerts.length || writes.length || errors.length || selected !== sample.input.options[0]) {
    throw new Error('Governance suspicion evaluation failed; inspect the recorded model result.');
  }
} finally { await browser.close(); }
