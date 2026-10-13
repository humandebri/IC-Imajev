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
for (const [extra, total] of [[48, 96], [49, 97]]) {
  const result = tokenizeInput(tokenizer, {
    ...votingInput, state: votingInput.state + " more".repeat(extra),
  }, readout.codes.map((entry) => entry.code));
  assert.equal(result.counts.total, total);
  assert.equal(result.counts.paidPrefix, 5);
  assert.equal(result.counts.paidSuffix, total - 5);
}
console.log("Five-token cache boundary and 96/97-token limits matched.");

for (const state of [null, {}, 1]) {
  assert.throws(() => tokenizeInput(tokenizer, { ...votingInput, state }, readout.codes.map(entry => entry.code)), /Context must be a string/);
}
