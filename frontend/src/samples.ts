import type { DecisionInput } from "./types";
import boomExamples from "./boom-examples.generated.json" with { type: "json" };

// Extreme input-only teaching examples. No model answers or reference labels.
export const samples: {
  id: string;
  label: string;
  description: string;
  input: DecisionInput;
}[] = [
  {
    id: "price-extreme-increase",
    label: "Value change",
    description: "Compare a $10 subscription with a $10,000 subscription",
    input: {
      state: "The monthly subscription price changes from $10 to $10,000.",
      question: "Did the price increase?",
      options: ["yes", "no"],
    },
  },
  {
    id: "delivery-missing-info",
    label: "Missing info",
    description: "Check delivery status without a due date or location",
    input: {
      state: "The package has shipped. The expected delivery date and its current location are unknown.",
      question: "Is the delivery late?",
      options: ["yes", "no"],
    },
  },
  {
    id: "review-extreme-mixed",
    label: "Three choices",
    description: "Classify a review with extremely positive and negative feedback",
    input: {
      state: "The product is perfect. The delivery was a disaster.",
      question: "What is the overall sentiment?",
      options: ["positive", "negative", "mixed"],
    },
  },
];

export interface RealWorldSample {
  id: string;
  label: string;
  description: string;
  sourceLabel: string;
  sourceUrl: string;
  input: DecisionInput;
  originalState: string;
  sourceStateSha256: string;
  tokens: number;
  audit: { kind: string; identifiers: { alias: string; value: string }[] };
}

// Build-generated exact compaction of the three retained original excerpts.
// Free input never passes through this transformation.
export const realWorldSamples: RealWorldSample[] = boomExamples;
