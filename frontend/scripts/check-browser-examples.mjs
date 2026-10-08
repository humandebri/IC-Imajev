import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { Tokenizer } from "@huggingface/tokenizers";
import { realWorldSamples } from "../src/samples.ts";
import { tokenizeInput } from "../src/tokenization.ts";
const original = JSON.parse(await readFile(new URL("../data/boom-examples.json", import.meta.url)));
const json = async name => JSON.parse(await readFile(new URL(`../public/tokenizer/${name}`, import.meta.url)));
const tokenizer = new Tokenizer(await json("tokenizer.json"), await json("tokenizer_config.json"));
const codes = (await json("decision_readout.json")).codes.map(entry => entry.code);
assert.equal(realWorldSamples.length, 3);
for (const sample of realWorldSamples) {
  const source = original.find(entry => entry.id === sample.id);
  assert.deepEqual(sample.input.options, source.input.options);
  assert.equal(sample.input.question, source.input.question);
  assert.equal(sample.originalState, source.input.state);
  assert.equal(sample.sourceStateSha256, createHash("sha256").update(source.input.state).digest("hex"));
  const counts = tokenizeInput(tokenizer, sample.input, codes).counts;
  assert.equal(counts.paidPrefix, 27);
  assert.equal(counts.total, sample.tokens);
  assert.ok(counts.total <= 96);
}
console.log("Three generated BOOM examples retain source evidence/question/options and fit the actual browser tokenizer budget.");
