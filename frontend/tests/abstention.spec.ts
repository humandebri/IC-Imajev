import { test, expect } from "@playwright/test";

test("abstention is explained separately and is not an editable option", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await expect(page.locator("#abstention-help")).toHaveText("Not enough information → Abstain");
  await expect(page.locator("fieldset.choices")).toHaveAttribute("aria-describedby", "abstention-help");
  await page.getByRole("button", { name: "Missing info" }).click();
  await expect(page.locator(".option-list input")).toHaveCount(2);
  await page.getByRole("button", { name: "Copy input" }).click();
  const copied = JSON.parse(await page.evaluate(() => navigator.clipboard.readText()));
  expect(copied.options).toEqual(["yes", "no"]);
  await expect(page.locator("#abstention-help")).toBeVisible();
});

test("measured abstention renders friendly wording separate from answer probabilities", async ({ page }) => {
  await page.goto("/");
  // Isolated component regression using the measured native delivery result.
  // Does not connect the public app, intercept an API or pre-fill its result.
  await page.evaluate(async () => {
    const load = (path: string) => import(/* @vite-ignore */ path);
    const [React, ReactDOM, components] = await Promise.all([
      load("/node_modules/.vite/deps/react.js"),
      load("/node_modules/.vite/deps/react-dom_client.js"),
      load("/src/ResultPanel.tsx"),
    ]);
    const host = document.createElement("div");
    host.id = "component-test";
    document.body.append(host);
    const react = React.default ?? React;
    const reactDOM = ReactDOM.default ?? ReactDOM;
    reactDOM.createRoot(host).render(react.createElement(components.ResultPanel, {
      options: ["yes", "no"],
      result: {
        value: null,
        probabilities: [0.0031993779052442942, 0.0036021575527132314],
        unknown_probability: 0.9931984645420425,
        abstained: true,
      },
    }));
  });
  const component = page.locator("#component-test");
  await expect(component.getByRole("heading", { name: "Not enough information" })).toBeVisible();
  await expect(component.locator(".result-label")).toHaveText("Abstained");
  await expect(component.locator(".abstention-probability")).toContainText("Not enough information");
  await expect(component.getByRole("meter", { name: "Model probability for not enough information" })).toHaveAttribute("value", "0.9931984645420425");
  await expect(component).not.toContainText("unknown");
});
