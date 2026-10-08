import { MAX_SUFFIX, MAX_TOKENS } from "../src/query-plan.ts";
import { test, expect } from "@playwright/test";
import { samples, realWorldSamples } from "../src/samples";

test("initial UI uses English document language and text", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.locator(".model-label")).toHaveText("Qwen 3.5-4B + adapter");
  await page.waitForFunction(() => {
    const image = document.querySelector<HTMLImageElement>(".kinic-logo");
    return image?.complete && image.naturalWidth > 0;
  });
  await expect(page.locator("body")).not.toContainText(
    /[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]/u,
  );
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "From text to decisions.",
  );
});

test("input samples and free editing do not produce results or inference requests", async ({
  page,
}) => {
  const inferenceRequests: string[] = [];
  page.on("request", (request) => {
    if (
      request.method() !== "GET" ||
      /\/api\/|\/canister\//.test(request.url())
    )
      inferenceRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Copy input" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Value change" }).click();
  await expect(page.getByLabel("Context")).toHaveValue(
    samples[0].input.state,
  );
  await page.getByRole("button", { name: "Missing info" }).click();
  await expect(page.getByLabel("Context")).toHaveValue(
    samples[1].input.state,
  );
  await page.getByRole("button", { name: "Three choices" }).click();
  await expect(
    page.getByRole("textbox", { name: "Option 3", exact: false }),
  ).toHaveValue("mixed");
  await expect(page.getByRole("button", { name: "Three choices" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByLabel("Question").fill("自由な質問");
  await expect(page.getByRole("button", { name: "Three choices" })).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.getByLabel("Context").fill("背景");
  await expect(
    page.getByRole("button", { name: "Copy input" }),
  ).toBeEnabled();
  await expect(page.getByRole("region", { name: "Result" })).toContainText(
    "Results appear here.",
  );
  await expect(page.locator("meter")).toHaveCount(0);
  await expect(
    page.getByRole("region", { name: "Result" }),
  ).not.toContainText(/\d+%/);
  expect(inferenceRequests).toEqual([]);
  await page.reload();
  await expect(page.getByLabel("Question")).toHaveValue("");
  await expect(page.getByLabel("Context")).toHaveValue("");
});

test("choice limits, empty question and whitespace duplicate validation", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Remove option 1", exact: true }),
  ).toBeDisabled();
  await page.getByLabel("Question").focus();
  await page.getByLabel("Context").focus();
  await expect(page.getByText("Enter a question.")).toBeVisible();
  await page.getByLabel("Question").fill("質問");
  for (let i = 0; i < 5; i++)
    await page.getByRole("button", { name: "Add option" }).click();
  await expect(
    page.getByRole("button", { name: "Add option" }),
  ).toBeDisabled();
  await expect(page.getByText("Enter an option.")).toHaveCount(5);
  for (let i = 3; i <= 7; i++)
    await page
      .getByRole("textbox", { name: `Option ${i}`, exact: false })
      .fill(`option${i}`);
  await expect(
    page.getByRole("button", { name: "Copy input" }),
  ).toBeEnabled();
  await page
    .getByRole("textbox", { name: "Option 2", exact: false })
    .fill(" yes ");
  await expect(page.getByText("Options must be unique.")).toHaveCount(2);
  await expect(
    page.getByRole("button", { name: "Copy input" }),
  ).toBeDisabled();
  await page.getByRole("textbox", { name: "Option 2", exact: false }).fill("no");
  await page
    .getByRole("button", { name: "Remove option 7", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Add option" }),
  ).toBeEnabled();
});

test("BOOM DAO excerpts include original values and a source link", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "BOOM #620" })).not.toBeVisible();
  await page.getByText("Real-world examples", { exact: true }).click();
  for (const sample of realWorldSamples) {
    await page.getByRole("button", { name: sample.label }).click();
    await expect(page.getByLabel("Context")).toHaveValue(sample.input.state);
    await expect(page.getByLabel("Question")).toHaveValue(sample.input.question);
    await expect(page.getByRole("link", { name: `View ${sample.sourceLabel}` })).toHaveAttribute("href", sample.sourceUrl);
    await expect(page.locator(".sample-source")).toContainText("Original payload excerpt");
    await expect(page.locator(".sample-source")).toContainText("Question added for this demo");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  }
  await page.getByRole("button", { name: "BOOM #617" }).click();
  await expect(page.getByLabel("Context")).toHaveValue(/86400[\s\S]*1728000000/);
  await page.getByRole("button", { name: "Value change" }).click();
  await expect(page.getByLabel("Context")).toHaveValue("The monthly subscription price changes from $10 to $10,000.");
  await expect(page.getByRole("button", { name: "BOOM #617" })).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator(".sample-source")).toHaveCount(0);
  await page.getByRole("button", { name: "BOOM #617" }).click();
  await page.getByLabel("Context").fill("Edited context");
  await expect(page.locator(".sample-source")).toHaveCount(0);
});

test("copy exports only UI input JSON", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  await page.getByRole("button", { name: "Copy input" }).click();
  await expect(page.locator(".copy-status")).toHaveText(
    "Input copied.",
  );
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(JSON.parse(copied)).toEqual(samples[0].input);
});

for (const width of [375, 768, 1440]) {
  test(`layout at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Decision input" }),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Result" })).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBeTruthy();
    await expect(page.locator(".token-counter dd")).toHaveCount(3, {
      timeout: 15000,
    });
    await page.screenshot({
      path: `test-results/playground-${width}.png`,
      fullPage: true,
    });
  });
}

test("keyboard can load samples, edit inputs and add choices", async ({
  page,
}) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "KINIC — Imajev on IC" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Value change" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Question")).toHaveValue(
    samples[0].input.question,
  );
  await page.getByRole("button", { name: "Add option" }).focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("textbox", { name: "Option 3", exact: false }),
  ).toBeVisible();
  await page.getByRole("textbox", { name: "Option 3", exact: false }).focus();
  await page.keyboard.type("maybe");
  await expect(
    page.getByRole("button", { name: "Copy input" }),
  ).toBeEnabled();
});

test("counts match actual pinned tokenizer and update for Japanese, quotes and empty state", async ({
  page,
}) => {
  const fixtures = (
    await import("./tokenizer-parity.json", { with: { type: "json" } })
  ).default;
  await page.goto("/");
  const counter = page.getByRole("region", { name: "Tokens", exact: true });
  for (const fixture of fixtures) {
    await page.getByLabel("Context").fill(fixture.input.state);
    await page.getByLabel("Question").fill(fixture.input.question);
    while (
      (await page.locator(".option-control input").count()) <
      fixture.input.options.length
    )
      await page.getByRole("button", { name: "Add option" }).click();
    while (
      (await page.locator(".option-control input").count()) >
      fixture.input.options.length
    )
      await page.locator(".remove-button").last().click();
    for (let i = 0; i < fixture.input.options.length; i++)
      await page
        .locator(".option-control input")
        .nth(i)
        .fill(fixture.input.options[i]);
    await expect(counter.locator("dd").nth(0)).toHaveText(
      String(fixture.token_ids.length),
      { timeout: 15000 },
    );
    await expect(counter.locator("dd").nth(1)).toHaveText(
      String(fixture.prefix),
    );
    await expect(counter.locator("dd").nth(2)).toHaveText(
      String(fixture.token_ids.length - fixture.prefix),
    );
  }
  await page.getByLabel("Context").fill("long ".repeat(100));
  await expect(counter).toContainText("limit exceeded");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
});

test("token details are available without cluttering the initial view", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".token-breakdown")).not.toBeVisible();
  await expect(page.locator(".candidate-preview")).toHaveCount(0);
  await page.getByText("Count details", { exact: true }).click();
  await expect(page.locator(".token-breakdown")).toBeVisible();
  await expect(page.locator(".token-counter dd")).toHaveCount(3);
});

test("former voting prefix uses the verified common-prefix token limit", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Question").fill("Approve?");
  await page.locator(".option-control input").nth(0).fill("no");
  await page.locator(".option-control input").nth(1).fill("yes");
  await page.getByText("Count details", { exact: true }).click();
  const counter = page.getByRole("region", { name: "Tokens", exact: true });
  const state = "Minimum voting dissolve delay changes from 1 day to 2 days.";
  await page.getByLabel("Context").fill(state + " more".repeat(MAX_TOKENS - 69));
  await expect(counter.locator(".token-counts dd")).toHaveText(String(MAX_TOKENS), { timeout: 15000 });
  await expect(counter).toContainText(`Query limit: ${MAX_SUFFIX} / ${MAX_SUFFIX}`);
  await expect(counter).not.toContainText("limit exceeded");
  await page.getByLabel("Context").fill(state + " more".repeat(MAX_TOKENS - 68));
  await expect(counter.locator(".token-counts dd")).toHaveText(String(MAX_TOKENS + 1));
  await expect(counter).toContainText(`Query limit: ${MAX_SUFFIX + 1} / ${MAX_SUFFIX}`);
  await expect(counter).toContainText("limit exceeded");
});
