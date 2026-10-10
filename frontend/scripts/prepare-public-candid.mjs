// Generate the published query subset from the exact agent wire schema.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { IDL } from "@icp-sdk/core/candid";
import { methods } from "../src/inference-agent.ts";
import release from "../src/inference-release.json" with { type: "json" };

const canonical = new URL("../../canisters/inference/public-query.did", import.meta.url);
const output = new URL("../public/api/public-query.did", import.meta.url);
// Keep an explicit allowlist so adding an owner API to the agent cannot
// silently publish it or mislabel an update method as a query.
const names = ["runAttentionInferenceStep", "runDeltaInferenceStep", "getModelStatus", "runInferenceStep", "runFinalInferenceStep", "getWeightCacheStatus"];
const declarations = names.map(name => {
  const schema = methods[name];
  return `  ${name} : ${IDL.Func(schema.args, schema.reply, ["query"]).display().replaceAll("→", "->").replaceAll("vec nat8", "blob")};`;
});
const text = `// Public browser inference query subset; owner/update APIs are excluded.
// Canister: ${release.canister}
// Module hash: ${release.module_hash}
// Opaque blobs require the five-token prefix frame/carry protocol, not plain text.
service : {
${declarations.join("\n")}
}
`;
if (process.argv.includes("--write")) await writeFile(canonical, text);
if (await readFile(canonical, "utf8") !== text) throw new Error("Public Candid differs from the agent schema. Run prepare-public-candid.mjs --write and review the change.");
if (!process.argv.includes("--check")) {
  await mkdir(new URL("../public/api/", import.meta.url), { recursive: true });
  await writeFile(output, text);
}
console.log(`Verified ${declarations.length} public query methods against the pinned agent schema.`);
