import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
const root = new URL("../../", import.meta.url);
const output = new URL("../public/tokenizer/", import.meta.url);
const lock = JSON.parse(
  await readFile(new URL("MODEL_LOCK.json", root), "utf8"),
);
await mkdir(output, { recursive: true });
const hashes = {};
for (const [component, name] of [
  ["base", "tokenizer.json"],
  ["base", "tokenizer_config.json"],
  ["adapter", "decision_readout.json"],
  ["base", "chat_template.jinja"],
]) {
  const file = lock.components[component].files[name];
  const bytes = await readFile(new URL(file.path, root));
  const hash = createHash("sha256").update(bytes).digest("hex");
  if (hash !== file.sha256)
    throw new Error(`MODEL_LOCK checksum mismatch: ${name}`);
  if (name !== "chat_template.jinja") {
    await writeFile(new URL(name, output), bytes);
    hashes[name] = hash;
  }
}
await writeFile(new URL("manifest.json", output), JSON.stringify({ hashes }));
console.log("Prepared pinned tokenizer assets (12.8 MB).");
