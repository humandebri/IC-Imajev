/** UI input only: not a Candid InferRequest or a tokenized prompt. */
export interface DecisionInput {
  state: string;
  question: string;
  options: string[];
}

/**
 * Real progress of an in-flight inference. Never synthesize these values:
 * - "steps": the client issues sequential inference queries and knows how
 *   many of the planned total have returned (e.g. 12 of 32). Weight/prefix
 *   preparation happens before release and is never counted here.
 * - "waiting": a single call (e.g. the paid update) whose internal progress
 *   is not observable; only elapsed time is shown.
 */
export type InferenceProgress =
  | {
      kind: "steps";
      completed: number;
      total: number;
      /** Date.now() when the first call was sent. */
      startedAt: number;
      /** Optional label; defaults to "判定中". */
      phase?: string;
    }
  | { kind: "waiting"; startedAt: number; phase?: string };

/** The decision fields to render once an actual API response is available. */
export interface DecisionResult {
  value: string | null;
  probabilities: number[];
  unknown_probability: number;
  abstained: boolean;
}
