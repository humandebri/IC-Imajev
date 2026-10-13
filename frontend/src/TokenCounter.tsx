import { MAX_SUFFIX, MAX_TOKENS } from "./query-plan.ts";
import { useEffect, useState } from "react";
import type { DecisionInput } from "./types";
import type { TokenCounts } from "./tokenization";
import { countTokens } from "./inference-client.ts";

export function TokenCounter({ input, onCounts }: { input: DecisionInput; onCounts?: (key: string, counts: TokenCounts | null) => void }) {
  const [answer, setAnswer] = useState<{
    key: string;
    counts?: TokenCounts;
    error?: string;
  } | null>(null);
  const inputKey = JSON.stringify(input);
  useEffect(() => {
    let active = true;
    onCounts?.(inputKey, null);
    countTokens(JSON.parse(inputKey)).then(counts => {
      if (active) { setAnswer({ key: inputKey, counts }); onCounts?.(inputKey, counts); }
    }).catch(error => { if (active) setAnswer({ key: inputKey, error: error.message }); });
    return () => { active = false; };
  }, [inputKey, onCounts]);
  const current = answer?.key === inputKey ? answer : null;
  const counts = current?.counts;
  const overLimit =
    counts?.paidSuffix !== null &&
    counts?.paidSuffix !== undefined &&
    counts.paidSuffix > MAX_SUFFIX;
  return (
    <section className="token-counter" aria-labelledby="tokens-heading">
      <h3 id="tokens-heading">Tokens</h3>
      <div role="status" aria-live="polite">
        {current?.error ? (
          <p className="error">{current.error}</p>
        ) : !counts ? (
          <p className="help">Loading tokenizer and counting…</p>
        ) : (
          <>
            <dl className="token-counts">
              <div>
                <dt>Total</dt>
                <dd key={counts.total}>{counts.total}</dd>
              </div>
            </dl>
          </>
        )}
      </div>
      <details className="token-details">
        <summary>Count details</summary>
        {counts && (
          <dl className="token-breakdown">
            <div>
              <dt>Shared prefix</dt>
              <dd>{counts.prefix}</dd>
            </div>
            <div>
              <dt>Input suffix</dt>
              <dd>{counts.suffix}</dd>
            </div>
          </dl>
        )}
        {counts && (
          <>
            {counts.paidPrefix === null ? (
              <p className="help">
                This input does not match the query API's fixed prefix.
              </p>
            ) : (
              <p className={overLimit ? "error" : "help"}>
                Query limit: {counts.paidSuffix} / {MAX_SUFFIX} additional tokens
                {overLimit ? " (limit exceeded)" : ""}.
                {` Fixed prefix: ${counts.paidPrefix} tokens; total limit: ${MAX_TOKENS} tokens.`}
              </p>
            )}
          </>
        )}
      </details>
    </section>
  );
}
