import type { Tokenizer } from "@huggingface/tokenizers";
import type { DecisionInput } from "./types.ts";

// Same instruction-free, JSON-quoted State layout as scripts/prepare_text.py.
const PREFIX5 = [248045, 846, 198, 1349, 25];
export interface TokenCounts {
  total: number;
  prefix: number;
  suffix: number;
  paidPrefix: number | null;
  paidSuffix: number | null;
}

export function renderPrompt(input: DecisionInput, codes: string[]): string {
  if (typeof input.state !== "string") throw new Error("Context must be a string.");
  // State is always a string in this UI; JSON quoting matches Python ensure_ascii=False.
  const choices = [...input.options, "unknown"];
  const prompt =
    `State: ${JSON.stringify(input.state)}\nQuestion: ${input.question}\n` +
    choices.map((text, i) => `${codes[i]}: ${text}`).join("\n");
  return `<|im_start|>user\n${prompt.trim()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

export function tokenizeInput(
  tokenizer: Tokenizer,
  input: DecisionInput,
  codes: string[],
) {
  const rendered = renderPrompt(input, codes);
  const tokenIds = tokenizer.encode(rendered, {
    add_special_tokens: false,
  }).ids;
  // Stop before the space/quote token, which can merge with the body.
  const prefix = PREFIX5.every((id, index) => tokenIds[index] === id) ? 5 : 0;
  const paidPrefix = prefix === 5 ? 5 : null;
  const counts: TokenCounts = {
    total: tokenIds.length,
    prefix,
    suffix: tokenIds.length - prefix,
    paidPrefix,
    paidSuffix: paidPrefix === null ? null : tokenIds.length - paidPrefix,
  };
  return { tokenIds, counts };
}
