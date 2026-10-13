import { test, expect } from "@playwright/test";

for (const [width, height] of [
  [1280, 720],
  [1440, 900],
  [1024, 768],
  [768, 1024],
  [375, 812],
  [390, 844],
]) {
  test(`initial English layout fits ${width}x${height} without scrolling`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    await expect(page.locator(".token-counts dd")).toHaveText("30", { timeout: 15000 });
    const dimensions = await page.evaluate(() => ({
      width: document.documentElement.scrollWidth,
      height: document.documentElement.scrollHeight,
    }));
    expect(dimensions.width).toBeLessThanOrEqual(width);
    expect(dimensions.height).toBeLessThanOrEqual(height);
    const header = (await page.locator(".site-header").boundingBox())!;
    const main = (await page.locator("main").boundingBox())!;
    const contentCenter = main.y + main.height / 2;
    const availableCenter = (header.y + header.height + height) / 2;
    expect(Math.abs(contentCenter - availableCenter)).toBeLessThan(1);
    for (const selector of [".input-panel", ".result-panel", ".token-counter"]) {
      const box = await page.locator(selector).boundingBox();
      expect(box).not.toBeNull();
      expect(box!.y + box!.height).toBeLessThanOrEqual(height);
    }
  });
}

test("short viewports keep content below the header and allow natural scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 600 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator(".token-counts dd")).toHaveText("30", { timeout: 15000 });
  const header = (await page.locator(".site-header").boundingBox())!;
  const main = (await page.locator("main").boundingBox())!;
  expect(main.y).toBeGreaterThanOrEqual(header.y + header.height);
  expect(await page.evaluate(() => document.documentElement.scrollHeight)).toBeGreaterThan(600);
  await page.getByRole("button", { name: "Copy input" }).scrollIntoViewIfNeeded();
  await expect(page.getByRole("button", { name: "Copy input" })).toBeInViewport();
});

test("expanded input remains accessible instead of being clipped", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/");
  await page.getByLabel("Context").fill("A long piece of context. ".repeat(100));
  await page.getByLabel("Question").fill("What should we decide?");
  for (let i = 0; i < 5; i++) {
    await page.getByRole("button", { name: "Add option" }).click();
    await page.getByRole("textbox", { name: `Option ${i + 3}`, exact: true }).fill(`answer ${i + 3}`);
  }
  await expect(page.getByRole("button", { name: "Copy input" })).toBeEnabled();
  await page.getByRole("button", { name: "Copy input" }).scrollIntoViewIfNeeded();
  await expect(page.getByRole("button", { name: "Copy input" })).toBeInViewport();
  await page.getByText("Count details", { exact: true }).click();
  await page.locator(".token-breakdown").scrollIntoViewIfNeeded();
  await expect(page.locator(".token-breakdown")).toBeVisible();
});
