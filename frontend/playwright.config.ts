import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:5173", browserName: "chromium" },
  webServer: {
    command:
      "node scripts/prepare-tokenizer.mjs && ../.venv/bin/python -B ../scripts/prepare_browser_prefix.py && ../.venv/bin/python -B ../scripts/prepare_browser_examples.py && node node_modules/vite/bin/vite.js --host 127.0.0.1",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
  },
});
