// Keep this entrypoint independent of npm's asset-generating pretest/prebuild.
import { spawnSync } from "node:child_process";

const checks = [
  ["node_modules/typescript/bin/tsc", "--noEmit"],
  ["scripts/prepare-public-candid.mjs", "--check"],
  ["scripts/check-query-plans.mjs"],
  ["scripts/check-query-profiles.mjs"],
  ["scripts/check-query-calibration-policy.mjs"],
  ["scripts/check-optimized-query-evidence.mjs"],
  ["scripts/check-release-prefix-assets.mjs"],
];
for (const args of checks) {
  console.log(`+ node --experimental-strip-types ${args.join(" ")}`);
  const result = spawnSync(process.execPath, ["--experimental-strip-types", ...args], {
    cwd: new URL("../", import.meta.url),
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
