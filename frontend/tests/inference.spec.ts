import { MAX_TOKENS } from "../src/query-plan.ts";
import { test, expect } from "@playwright/test";

// UI lifecycle tests inject the client boundary; protocol and real browser
// networking are verified separately against actual mainnet evidence.
async function clientFixture(page: import("@playwright/test").Page) {
  await page.route("**/src/inference-client.ts*", route => route.fulfill({
    contentType: "text/javascript", body: `
      export async function countTokens(input) {
        const total = input.state.length > 200 ? ${MAX_TOKENS + 1} : 64;
        return {total, prefix:27, suffix:total-27, paidPrefix:27, paidSuffix:total-27};
      }
      export function infer(input, progress) {
        const promise = new Promise((resolve, reject) => {
          let completed = 0;
          const timer = setInterval(() => {
            progress(++completed, 63);
            if (input.question === "gateway unavailable") {
              clearInterval(timer);
              reject(new Error("IC gateway busy (503). Please try again."));
              return;
            }
            if(completed===63) {
              clearInterval(timer);
              resolve({ value:input.options[0], probabilities:input.options.map((_,i)=>i===0?0.8:0.1/(input.options.length-1)), unknown_probability:0.1, abstained:false });
            }
          }, 40);
        });
        // Deliberately allow an old completion after cancellation. The UI
        // must fence its result even if an underlying request cannot stop.
        return {promise, cancel() {}};
      }
    `,
  }));
}

test("run displays real progress and preserves submitted option labels while editing", async ({ page }) => {
  await clientFixture(page); await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  await expect(page.getByRole("button", { name: "Run inference" })).toBeEnabled();
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuemax", "63");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
  await page.locator(".option-control input").first().fill("edited option");
  await expect(page.locator(".result-panel")).toHaveAttribute("data-state", "done");
  await expect(page.locator(".actual-result h3")).toHaveText("yes · 80%");
  await expect(page.locator(".result-meta")).toContainText("Runner-up optionno · 10%");
  await expect(page.locator(".result-meta")).toContainText(/Completed in\d+\.\d s/);
  await expect(page.locator(".probabilities li").first()).toContainText("yes");
  await expect(page.locator(".probabilities")).not.toContainText("edited option");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeEnabled();
});

test("cancel and rerun discard the previous late completion", async ({ page }) => {
  await clientFixture(page); await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  await page.getByRole("button", { name: "Run inference" }).click();
  const cancel = page.getByRole("button", { name: "Cancel", exact: true });
  await cancel.focus();
  await cancel.press("Enter");
  await expect(page.getByRole("alert")).toHaveText("Inference cancelled.");
  await page.locator(".option-control input").first().fill("new option");
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.locator(".actual-result h3")).toHaveText("new option · 80%");
  await expect(page.locator(".result-meta")).toContainText("Runner-up optionno · 10%");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("over-limit input cannot start inference", async ({ page }) => {
  await clientFixture(page); await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  await page.getByLabel("Context").fill("long ".repeat(60));
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
  await expect(page.locator(".token-counter")).toContainText("limit exceeded");
});

test("query failure clears progress and allows the user to retry", async ({ page }) => {
  await clientFixture(page); await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  await page.getByLabel("Question", { exact: false }).fill("gateway unavailable");
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.getByRole("alert")).toHaveText("IC gateway busy (503). Please try again.");
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  await expect(page.locator(".actual-result")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Run inference" })).toBeEnabled();
});

test("reserved abstention option blocks submission before any canister request", async ({ page }) => {
  let requests = 0;
  await page.route("https://icp-api.io/**", route => { requests++; return route.abort(); });
  await page.goto("/");
  await page.getByRole("button", { name: "Value change" }).click();
  const run = page.getByRole("button", { name: "Run inference" });
  await expect(run).toBeEnabled();
  for (const value of ["__unknown__", "  __unknown__  "]) {
    await page.getByLabel("Option 1", { exact: true }).fill(value);
    await expect(page.getByText("This option is reserved for abstention.")).toBeVisible();
    await expect(page.getByLabel("Option 1", { exact: true })).toHaveAttribute("aria-invalid", "true");
    await expect(run).toBeDisabled();
    // Even programmatic form submission must obey the validation guard.
    await page.locator("form").evaluate(form => (form as HTMLFormElement).requestSubmit());
    await expect(page.locator(".result-panel")).toHaveAttribute("data-state", "idle");
    expect(requests).toBe(0);
  }
  await page.getByLabel("Option 1", { exact: true }).fill("yes");
  await expect(run).toBeEnabled();
  await expect(page.getByText("This option is reserved for abstention.")).toHaveCount(0);
});
