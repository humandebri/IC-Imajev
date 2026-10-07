import type { DecisionInput } from "./types";

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

const proposalUrl = (id: number) =>
  `https://sns-api.internetcomputer.org/api/v1/snses/xjngq-yaaaa-aaaaq-aabha-cai/proposals/${id}`;

/**
 * Verbatim excerpts of payload_text_rendering from the BOOM DAO SNS API.
 * Titles come from proposal_title. Questions are demo questions, not proposer
 * text. Labels explain the interaction; proposal numbers belong to provenance.
 * No model answers or reference labels are included.
 */
export const realWorldSamples: {
  id: string;
  label: string;
  description: string;
  sourceLabel: string;
  sourceUrl: string;
  input: DecisionInput;
}[] = [
  {
    id: "boom-620",
    label: "BOOM #620",
    description: "Compare the previous and proposed voting lock duration",
    sourceLabel: "BOOM DAO #620",
    sourceUrl: proposalUrl(620),
    input: {
      state: `SNS parameters adjustment

# Proposal to change nervous system parameters:
## Current nervous system parameters:
    neuron_minimum_dissolve_delay_to_vote_seconds: Some(
        86400,
    ),
## New nervous system parameters:
    neuron_minimum_dissolve_delay_to_vote_seconds: Some(
        172800,
    ),`,
      question: "Does this proposal increase the minimum lock duration needed for voting?",
      options: ["yes", "no"],
    },
  },
  {
    id: "boom-653",
    label: "BOOM #653",
    description: "A mint amount alone cannot establish the recipient's share of total supply",
    sourceLabel: "BOOM DAO #653",
    sourceUrl: proposalUrl(653),
    input: {
      state: `SNS Adjustment

# Proposal to mint SNS Tokens:
## Amount: 250000000.00000000 SNS Tokens
## Amount (e8s): 25000000000000000
## Target principal: 4mpab-ayg5c-m2frb-vjja5-b2qbe-5rxgv-33ova-ddjgm-egfav-br5hu-gqe
## Target account: 4mpab-ayg5c-m2frb-vjja5-b2qbe-5rxgv-33ova-ddjgm-egfav-br5hu-gqe
## Memo: 0`,
      question: "Will the recipient own more than half of the total token supply after this mint?",
      options: ["yes", "no"],
    },
  },
  {
    id: "boom-617",
    label: "BOOM #617",
    description: "Classify the change in the minimum voting lock duration",
    sourceLabel: "BOOM DAO #617",
    sourceUrl: proposalUrl(617),
    input: {
      state: `SNS parameters adjustment

# Proposal to change nervous system parameters:
## Current nervous system parameters:
    neuron_minimum_dissolve_delay_to_vote_seconds: Some(
        86400,
    ),
    max_dissolve_delay_seconds: Some(
        2629800,
    ),
## New nervous system parameters:
    neuron_minimum_dissolve_delay_to_vote_seconds: Some(
        1728000000,
    ),
    max_dissolve_delay_seconds: Some(
        2629800000,
    ),`,
      question: "How does the minimum lock duration needed for voting change?",
      options: ["increases", "decreases", "unchanged"],
    },
  },
];
