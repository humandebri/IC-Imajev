import { Tokenizer } from "@huggingface/tokenizers";
import { tokenizeInput } from "./tokenization.ts";
import type { DecisionInput } from "./types";

async function loadTokenizer() {
  const base = `${import.meta.env.BASE_URL}tokenizer/`;
  const manifestResponse = await fetch(`${base}manifest.json`);
  if (!manifestResponse.ok)
    throw new Error("Could not load tokenizer settings.");
  const manifest = (await manifestResponse.json()) as {
    hashes: Record<string, string>;
  };
  async function load(name: string) {
    const response = await fetch(`${base}${name}`);
    if (!response.ok)
      throw new Error("Could not load a tokenizer file.");
    const buffer = await response.arrayBuffer();
    const hash = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
      (n) => n.toString(16).padStart(2, "0"),
    ).join("");
    if (hash !== manifest.hashes[name])
      throw new Error("Tokenizer file verification failed.");
    return JSON.parse(new TextDecoder().decode(buffer));
  }
  const [json, config, readout] = await Promise.all([
    load("tokenizer.json"),
    load("tokenizer_config.json"),
    load("decision_readout.json"),
  ]);
  return {
    tokenizer: new Tokenizer(json, config),
    codes: readout.codes.map(
      (entry: { code: string }) => entry.code,
    ) as string[],
  };
}
let loaded: ReturnType<typeof loadTokenizer> | undefined;
self.onmessage = async (
  event: MessageEvent<{ id: number; input: DecisionInput }>,
) => {
  const { id, input } = event.data;
  try {
    const { tokenizer, codes } = await (loaded ??= loadTokenizer());
    self.postMessage({
      id,
      counts: tokenizeInput(tokenizer, input, codes).counts,
    });
  } catch (error) {
    self.postMessage({
      id,
      error:
        error instanceof Error
          ? error.message
          : "Could not count tokens.",
    });
  }
};
