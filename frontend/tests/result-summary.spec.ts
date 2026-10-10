import { test, expect, type Page } from "@playwright/test";

// Exercise the real form, submitted option snapshot and result layout with
// controlled responses. No mainnet calls or model accuracy claims.
async function showResult(
  page: Page,
  options: string[],
  probabilities: number[],
  selectedIndex: number | null,
  unknownProbability: number,
) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route("**/src/inference-client.ts*", route => route.fulfill({
    contentType: "text/javascript",
    body: `
      export async function countTokens() {
        return {total:64, prefix:5, suffix:59, paidPrefix:5, paidSuffix:59};
      }
      export function infer(input, progress) {
        const selectedIndex = ${JSON.stringify(selectedIndex)};
        return {cancel() {}, promise: new Promise(resolve => setTimeout(() => {
          progress(1, 1);
          resolve({value: selectedIndex === null ? null : input.options[selectedIndex],
            probabilities: ${JSON.stringify(probabilities)},
            unknown_probability: ${unknownProbability}, abstained: selectedIndex === null});
        }, 50))};
      }
    `,
  }));
  await page.goto("/");
  await page.getByLabel("Question", { exact: false }).fill("Which option?");
  for (let index = 0; index < options.length; index++) {
    if (index >= 2) await page.getByRole("button", { name: "Add option" }).click();
    await page.getByLabel(`Option ${index + 1}`, { exact: true }).fill(options[index]);
  }
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.locator(".result-panel")).toHaveAttribute("data-state", "done");
}

for (const scenario of [
  { name: "third candidate wins", options: ["A", "B", "C"], probabilities: [0.12, 0.23, 0.6], selected: 2, unknown: 0.05, heading: "C · 60%", alternative: "Runner-up optionB · 23%" },
  { name: "four candidates with runner-up after winner", options: ["A", "B", "C", "D"], probabilities: [0.05, 0.5, 0.3, 0.1], selected: 1, unknown: 0.05, heading: "B · 50%", alternative: "Runner-up optionC · 30%" },
  { name: "abstention keeps candidate ranking separate", options: ["A", "B", "C"], probabilities: [0.05, 0.1, 0.2], selected: null, unknown: 0.65, heading: "Not enough information · 65%", alternative: "Top optionC · 20%" },
  { name: "tied alternatives retain input order", options: ["A", "B", "C"], probabilities: [0.2, 0.5, 0.2], selected: 1, unknown: 0.1, heading: "B · 50%", alternative: "Runner-up optionA · 20%" },
]) {
  test(scenario.name, async ({ page }) => {
    await showResult(page, scenario.options, scenario.probabilities, scenario.selected, scenario.unknown);
    await expect(page.locator(".actual-result h3")).toHaveText(scenario.heading);
    await expect(page.locator(".result-meta > div").first()).toHaveText(scenario.alternative);
    await expect(page.locator(".result-meta")).toContainText(/Completed in\d+\.\d s/);
  });
}

for (const width of [320, 375, 390]) {
  for (const language of ["ascii", "japanese"]) {
    test(`long ${language} candidates fit ${width}px`, async ({ page }, testInfo) => {
      await page.setViewportSize({ width, height: 812 });
      const winner = language === "ascii" ? "A".repeat(128) : "判".repeat(42);
      const runnerUp = language === "ascii" ? "B".repeat(128) : "候".repeat(42);
      await showResult(page, ["Other", runnerUp, winner], [0.1, 0.25, 0.6], 2, 0.05);
      await expect(page.locator(".actual-result h3")).toHaveText(`${winner} · 60%`);
      await expect(page.locator(".result-meta > div").first()).toHaveText(`Runner-up option${runnerUp} · 25%`);
      const layout = await page.locator(".result-panel").evaluate(panel => {
        const bounds = panel.getBoundingClientRect();
        const clipped: string[] = [];
        const walker = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
          const node = walker.currentNode;
          if (!node.textContent?.trim()) continue;
          const range = document.createRange();
          range.selectNodeContents(node);
          for (const rect of range.getClientRects()) {
            if (rect.left < bounds.left - 1 || rect.right > bounds.right + 1 ||
                rect.top < bounds.top - 1 || rect.bottom > bounds.bottom + 1) {
              clipped.push(node.textContent);
              break;
            }
          }
        }
        return { clipped, pageWidth: document.documentElement.scrollWidth,
          panelLeft: bounds.left, panelRight: bounds.right };
      });
      expect(layout.clipped).toEqual([]);
      expect(layout.pageWidth).toBeLessThanOrEqual(width);
      expect(layout.panelLeft).toBeGreaterThanOrEqual(0);
      expect(layout.panelRight).toBeLessThanOrEqual(width);
      const percentageLines = await page.locator(".probabilities strong").evaluateAll(elements =>
        elements.map(element => {
          const range = document.createRange();
          range.selectNodeContents(element);
          return range.getClientRects().length;
        }),
      );
      expect(percentageLines).toEqual([1, 1, 1, 1]);
      await page.locator(".result-panel").screenshot({ path: testInfo.outputPath("result.png") });
    });
  }
}
