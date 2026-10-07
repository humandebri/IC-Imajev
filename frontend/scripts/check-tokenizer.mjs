import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { Tokenizer } from "@huggingface/tokenizers";
import { tokenizeInput } from "../src/tokenization.ts";
const json = (name) =>
  readFile(new URL(name, import.meta.url), "utf8").then(JSON.parse);
const [model, config, readout, fixtures] = await Promise.all([
  json("../public/tokenizer/tokenizer.json"),
  json("../public/tokenizer/tokenizer_config.json"),
  json("../public/tokenizer/decision_readout.json"),
  json("../tests/tokenizer-parity.json"),
]);
const tokenizer = new Tokenizer(model, config);
for (const fixture of fixtures) {
  const result = tokenizeInput(
    tokenizer,
    fixture.input,
    readout.codes.map((entry) => entry.code),
  );
  assert.deepEqual(result.tokenIds, fixture.token_ids);
  assert.equal(result.counts.prefix, fixture.prefix);
}
console.log(
  `Python TextPreparer parity: ${fixtures.length} full token-ID sequences matched.`,
);
