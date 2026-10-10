// Explicit read-only mainnet smoke test of the enriched real-world examples.
import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
const browser = await chromium.launch();
const directory = new URL(process.env.BOOM_TEST_OUTPUT ?? '../../artifacts/boom-enriched-demo-20261009/final/', import.meta.url);
await mkdir(directory, { recursive: true });
const cases = process.env.BOOM_TEST_MODE === 'controls' ? [
  { id: '584-metadata-control', sampleId: '584', expected: 'metadata update',
    state: 'Title: SNS Metadata Adjustment. Payload: update the DAO logo; no funds are moved.' },
  { id: '584-neutral-title-control', sampleId: '584', expected: 'treasury transfer',
    state: 'Title: SNS Adjustment. Payload: send 20M BOOM from DAO treasury to P1.' },
] : (process.env.BOOM_TEST_IDS ?? '584,653').split(',').map(id => ({
  id, sampleId: id, expected: id === '584' ? 'treasury transfer' : 'no',
}));
try {
  const results = await Promise.all(cases.map(async testCase => {
    const { id, sampleId, expected } = testCase;
    const page = await browser.newPage();
    const writes = [], errors = [];
    page.on('request', request => { if (/\/call(?:\/|$)/.test(request.url())) writes.push(request.url()); });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.PLAYGROUND_URL ?? 'http://127.0.0.1:5173');
    await page.getByText('Real-world examples', { exact: true }).click();
    await page.getByRole('button', { name: `BOOM #${sampleId}` }).click();
    if (testCase.state) await page.getByLabel('Context').fill(testCase.state);
    await page.waitForFunction(() => !document.querySelector('.primary-button').disabled, null, { timeout: 30000 });
    await page.getByRole('button', { name: 'Run inference' }).click();
    await page.waitForFunction(() => document.querySelector('.result-panel[data-state="done"]') || document.querySelector('[role="alert"]'), null, { timeout: 360000 });
    const alerts = await page.locator('[role="alert"]').allTextContents();
    const result = { id, expected, syntheticControl: Boolean(testCase.state), alerts, writes, errors,
      state: await page.getByLabel('Context').inputValue(), question: await page.getByLabel('Question').inputValue(),
      selected: alerts.length ? null : (await page.locator('.actual-result h3').innerText()).replace(/\s*·.*$/, ''),
      result: await page.locator('.result-panel').innerText() };
    await page.screenshot({ path: new URL(`${id}.png`, directory).pathname, fullPage: true });
    await page.close();
    console.log(JSON.stringify(result));
    return result;
  }));
  await writeFile(new URL('report.json', directory), JSON.stringify(results, null, 2) + '\n');
  if (results.some(result => result.alerts.length || result.writes.length || result.errors.length || result.selected !== result.expected)) {
    throw new Error('Enriched demo did not select expected action without writes/errors; inspect report.');
  }
} finally { await browser.close(); }
