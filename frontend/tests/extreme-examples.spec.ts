import { test, expect } from "@playwright/test";

const cases = [
  {
    label: "Value change",
    context: "The monthly subscription price changes from $10 to $10,000.",
    question: "Did the price increase?",
    options: ["yes", "no"],
    total: 74,
    suffix: 47,
  },
  {
    label: "Missing info",
    context: "The package has shipped. The expected delivery date and its current location are unknown.",
    question: "Is the delivery late?",
    options: ["yes", "no"],
    total: 72,
    suffix: 45,
  },
  {
    label: "Three choices",
    context: "The product is perfect. The delivery was a disaster.",
    question: "What is the overall sentiment?",
    options: ["positive", "negative", "mixed"],
    total: 72,
    suffix: 45,
  },
];

for (const scenario of cases) {
  test(`extreme ${scenario.label} input and real token count`, async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: scenario.label }).click();
    await expect(page.getByLabel("Context")).toHaveValue(scenario.context);
    await expect(page.getByLabel("Question")).toHaveValue(scenario.question);
    for (const [i, option] of scenario.options.entries()) {
      await expect(page.getByRole("textbox", { name: `Option ${i + 1}`, exact: true })).toHaveValue(option);
    }
    await expect(page.locator(".token-counts dd")).toHaveText(String(scenario.total), { timeout: 15000 });
    await page.getByText("Count details", { exact: true }).click();
    await expect(page.locator(".token-breakdown dd").last()).toHaveText(String(scenario.suffix));
    await expect(page.locator(".token-counter")).not.toContainText("limit exceeded");
    await expect(page.locator(".actual-result")).toHaveCount(0);
    await expect(page.locator(".sample-source")).toHaveCount(0);
  });
}
