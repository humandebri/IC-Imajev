import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { test, expect } from "@playwright/test";

// Use a real HTTP server: Playwright routing disables the browser HTTP cache.
for (const corrupted of ["manifest.json", "layer-00.bin"]) {
  test(`recovers immutable HTTP cache corruption in ${corrupted}`, async ({ page, baseURL }) => {
    const assets = new URL("../public/inference/prefix27-v1/", import.meta.url);
    const manifestBytes = await readFile(new URL("manifest.json", assets));
    const manifest = JSON.parse(manifestBytes.toString());
    const files = new Map<string, Buffer>([["manifest.json", manifestBytes]]);
    for (const entry of Object.values(manifest.assets) as { file: string }[]) {
      files.set(entry.file, await readFile(new URL(entry.file, assets)));
    }
    let healthy = false, targetRequests = 0;
    const server = createServer((request, response) => {
      const name = request.url!.split("/").at(-1)!;
      if (!files.has(name)) {
        response.writeHead(200, { "Content-Type": "text/html" });
        response.end("<!doctype html><title>Prefix HTTP cache fixture</title>");
        return;
      }
      if (name === corrupted) targetRequests++;
      response.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Cache-Control": "public, max-age=31536000, immutable",
      });
      response.end(name === corrupted && !healthy ? Buffer.alloc(files.get(name)!.length) : files.get(name));
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    const address = server.address() as { port: number };
    const base = `http://127.0.0.1:${address.port}/prefix/`;
    try {
      await page.goto(base);
      // Prime an invalid 200 response into the real immutable browser cache.
      await page.evaluate(async url => { await (await fetch(url)).arrayBuffer(); }, `${base}${corrupted}`);
      expect(targetRequests).toBe(1);
      healthy = true;
      const result = await page.evaluate(async ({ base, moduleUrl }) => {
        const { loadPrefix } = await import(/* @vite-ignore */ moduleUrl);
        const controller = new AbortController();
        try {
          const prefix = await loadPrefix(base, controller.signal);
          const bytes = await prefix.asset(0);
          return { bytes: bytes.length, digest: Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
            b => b.toString(16).padStart(2, "0")).join("") };
        } finally { controller.abort(); }
      }, { base, moduleUrl: `${baseURL}/src/prefix-assets.ts` });
      expect(result.bytes).toBe(manifest.assets["0"].bytes);
      expect(result.digest).toBe(manifest.assets["0"].sha256);
      expect(targetRequests).toBe(2); // One priming request and one cache-bypass repair.
      const healedDigest = await page.evaluate(async url => {
        const bytes = await (await fetch(url)).arrayBuffer();
        return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
          b => b.toString(16).padStart(2, "0")).join("");
      }, `${base}${corrupted}`);
      const { createHash } = await import("node:crypto");
      expect(healedDigest).toBe(createHash("sha256").update(files.get(corrupted)!).digest("hex"));
      expect(targetRequests).toBe(2); // Reload mode replaced the invalid cached response.
    } finally {
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
    }
  });
}
