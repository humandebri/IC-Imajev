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

const votingInput = {
  state: "Minimum voting dissolve delay changes from 1 day to 2 days.",
  question: "Approve?",
  options: ["no", "yes"],
};
for (const [extra, total] of [[15, 84], [16, 85]]) {
  const result = tokenizeInput(tokenizer, {
    ...votingInput, state: votingInput.state + " more".repeat(extra),
  }, readout.codes.map((entry) => entry.code));
  assert.deepEqual(result.tokenIds.slice(27, 38),
    [27756,15209,70173,7383,4203,494,220,16,1834,310,220]);
  assert.equal(result.counts.total, total);
  assert.equal(result.counts.paidPrefix, 27);
  assert.equal(result.counts.paidSuffix, total - 27);
}
console.log("Former voting prefix counts as suffix; 84/85-token boundaries matched.");
