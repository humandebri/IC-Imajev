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
  originalQuestion: string;
  sourceStateSha256: string;
  tokens: number;
  audit: { kind: string; identifiers: { alias: string; value: string }[];
    participation_snapshot?: {
      sourceUrl: string; sourceKind: string; startedAt: string; completedAt: string;
      referenceTimestampSeconds: number; totalNeurons: number; sha256: string; scenario: string; semantics: string;
      analysis: { before: { count: number; largest_share_bps_floor: number | null };
        after: { count: number; largest_share_bps_floor: number | null };
        excluded_neurons: number; excluded_percent_bps_floor: number | null };
    };
    derived?: { share_percent_bounds: number[]; formula: string };
    ledger_snapshot?: { ledgerCanisterId: string; recipient: string; startedAt: string; completedAt: string;
      decimals: number; totalSupplyE8s: string; recipientBalanceE8s: string; scenario: string };
  };
}

// Build-generated proposal excerpts; ledger evidence uses enclosing numeric ranges.
// Free input never passes through this transformation.
export const realWorldSamples: RealWorldSample[] = boomExamples;
