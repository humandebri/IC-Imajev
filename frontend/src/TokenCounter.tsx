import { useEffect, useRef, useState } from "react";
import type { DecisionInput } from "./types";
import type { TokenCounts } from "./tokenization";

export function TokenCounter({ input }: { input: DecisionInput }) {
  const worker = useRef<Worker | null>(null);
  const latest = useRef(0);
  const [answer, setAnswer] = useState<{
    key: string;
    counts?: TokenCounts;
    error?: string;
  } | null>(null);
  const inputKey = JSON.stringify(input);
  const requestKey = useRef("");
  useEffect(() => {
    const instance = new Worker(
      new URL("./tokenizer.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = instance;
    instance.onmessage = (event) => {
      if (event.data.id === latest.current)
        setAnswer({
          key: requestKey.current,
          counts: event.data.counts,
          error: event.data.error,
        });
    };
    instance.onerror = () =>
      setAnswer({
        key: requestKey.current,
        error:
          "Could not start the tokenizer. Reload the page.",
      });
    return () => {
      worker.current = null;
      instance.terminate();
    };
  }, []);
  useEffect(() => {
    requestKey.current = inputKey;
    worker.current?.postMessage({
      id: ++latest.current,
      input: JSON.parse(inputKey),
    });
  }, [inputKey]);
  const current = answer?.key === inputKey ? answer : null;
  const counts = current?.counts;
  const overLimit =
    counts?.paidSuffix !== null &&
    counts?.paidSuffix !== undefined &&
    counts.paidSuffix > 57;
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
                This input does not match the paid update API's fixed prefix.
              </p>
            ) : (
              <p className={overLimit ? "error" : "help"}>
                Paid update API: {counts.paidSuffix} / 57 additional tokens
                {overLimit ? " (limit exceeded)" : ""}.
                {counts.paidPrefix === 38
                  ? " Prepare the 38-token prefix first."
                  : ""}
              </p>
            )}
          </>
        )}
      </details>
    </section>
  );
}
