import type { Tokenizer } from "@huggingface/tokenizers";
import type { DecisionInput } from "./types.ts";

// Same text-only-short-v2 layout as scripts/prepare_text.py, one presentation order.
const INSTRUCTIONS =
  "Inspect the available evidence and answer the question using the stated criteria. Return only the single option code.";
const PREFIX27 = [
  248045, 846, 198, 56555, 279, 2420, 5721, 321, 4087, 279, 3296, 1608, 279,
  10661, 12521, 13, 3301, 1132, 279, 3074, 2904, 1970, 13, 198, 1349, 25, 328,
];
export interface TokenCounts {
  total: number;
  prefix: number;
  suffix: number;
  paidPrefix: number | null;
  paidSuffix: number | null;
}

export function renderPrompt(input: DecisionInput, codes: string[]): string {
  // State is always a string in this UI; JSON quoting matches Python ensure_ascii=False.
  const choices = [...input.options, "unknown"];
  const prompt =
    `${INSTRUCTIONS}\nState: ${JSON.stringify(input.state)}\nQuestion: ${input.question}\n` +
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
  const prefixText = rendered.split("\nState: ")[0] + '\nState: "';
  const prefixIds = tokenizer.encode(prefixText, {
    add_special_tokens: false,
  }).ids;
  let prefix = 0;
  while (prefix < prefixIds.length && prefixIds[prefix] === tokenIds[prefix])
    prefix++;
  const matches = (ids: number[]) =>
    ids.every((id, index) => tokenIds[index] === id);
  const paidPrefix = matches(PREFIX27) ? 27 : null;
  const counts: TokenCounts = {
    total: tokenIds.length,
    prefix,
    suffix: tokenIds.length - prefix,
    paidPrefix,
    paidSuffix: paidPrefix === null ? null : tokenIds.length - paidPrefix,
  };
  return { tokenIds, counts };
}
