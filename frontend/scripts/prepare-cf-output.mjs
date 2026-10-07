import path from "node:path";
import { InputConfigSchema } from "@cloudflare/config";
import { cleanBuildOutputDir, writeAssets, writeRootConfig, writeWorkerConfig } from "@cloudflare/build-output-utils";
import config from "../cloudflare.config.ts";

// cf deploy --prebuilt consumes the static Vite build without framework migration.
const root = path.resolve(import.meta.dirname, "..");
const parsed = InputConfigSchema.parse(config);
await cleanBuildOutputDir(root);
await writeRootConfig(root, parsed.settings, { isPreview: false });
await writeWorkerConfig({ root, config: parsed.worker });
await writeAssets({ root, sourceDirectory: path.join(root, "dist") });
console.log("Prepared static assets for cf deploy --prebuilt.");
