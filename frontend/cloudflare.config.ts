import { defineConfig } from "cf/config";

export default defineConfig({
	worker: {
		name: "ic-imajev",
		workersDev: true,
		compatibilityDate: "2026-10-01",
		assets: {
			notFoundHandling: "single-page-application",
		},
		domains: [
			"imajev.kinic.xyz",
		],
	},
});
