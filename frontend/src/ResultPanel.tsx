import type { CSSProperties } from "react";
import { InferenceStatus } from "./InferenceStatus";
import type { DecisionResult, InferenceProgress } from "./types";

const percent = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);

/** Purely decorative: one input fanning out to one branch per current option. */
function DecisionPrism({ count }: { count: number }) {
  const n = Math.max(2, Math.min(7, count));
  const top = 34;
  const bottom = 166;
  const ys = Array.from({ length: n }, (_, i) =>
    n === 1 ? 100 : top + ((bottom - top) * i) / (n - 1),
  );
  return (
    <div className="prism" aria-hidden="true">
      <svg viewBox="0 0 320 200">
        <defs>
          <linearGradient
            id="prism-stroke"
            gradientUnits="userSpaceOnUse"
            x1="58"
            x2="270"
            y1="0"
            y2="0"
          >
            <stop offset="0" stopColor="var(--glow-a)" stopOpacity="0.95" />
            <stop offset="1" stopColor="var(--glow-b)" stopOpacity="0.55" />
          </linearGradient>
          <radialGradient id="prism-core">
            <stop offset="0" stopColor="#fff" />
            <stop offset="0.45" stopColor="var(--glow-a)" />
            <stop offset="1" stopColor="var(--glow-a)" stopOpacity="0" />
          </radialGradient>
        </defs>
        {ys.map((y, i) => (
          <path
            key={i}
            className="prism-branch"
            style={
              {
                d: `path("M58 100 C 170 100 168 ${y} 270 ${y}")`,
                animationDelay: `${i * -0.35}s`,
              } as CSSProperties
            }
            d={`M58 100 C 170 100 168 ${y} 270 ${y}`}
            stroke="url(#prism-stroke)"
          />
        ))}
        {ys.map((y, i) => (
          <circle
            key={i}
            className="prism-node"
            style={{ cy: y } as CSSProperties}
            cx="270"
            cy={y}
            r="5"
          />
        ))}
        <circle className="prism-halo" cx="58" cy="100" r="34" fill="url(#prism-core)" />
        <circle className="prism-source" cx="58" cy="100" r="9" />
      </svg>
    </div>
  );
}

/** Only pass an actual response and the options from that same request. */
export function ResultPanel({
  result,
  options,
  progress = null,
  elapsedMs,
}: {
  result: DecisionResult | null;
  options: string[];
  /** Pass only real progress from in-flight calls; null when idle. */
  progress?: InferenceProgress | null;
  /** End-to-end duration of this completed run, including preparation and network. */
  elapsedMs?: number;
}) {
  const running = result === null && progress !== null;
  const selectedProbability = result === null ? undefined : result.abstained
    ? result.unknown_probability
    : result.probabilities[options.indexOf(result.value ?? "")];
  const alternative = result === null ? undefined : options
    .map((option, index) => ({ option, probability: result.probabilities[index] }))
    .filter(candidate => result.abstained || candidate.option !== result.value)
    .sort((a, b) => b.probability - a.probability)[0];
  return (
    <section
      className="result-panel"
      aria-labelledby="result-heading"
      aria-busy={running}
      data-state={result !== null ? "done" : running ? "running" : "idle"}
    >
      <div className="panel-heading">
        <h2 id="result-heading">Result</h2>
        <span className="status-dot" aria-hidden="true" />
      </div>
      {running ? (
        <div className="result-empty is-running">
          <DecisionPrism count={options.length} />
          <InferenceStatus progress={progress} />
        </div>
      ) : result === null ? (
        <div className="result-empty">
          <DecisionPrism count={options.length} />
          <p>Results appear here.</p>
        </div>
      ) : (
        <div className="actual-result" aria-live="polite">
          <p className="result-label">
            {result.abstained ? "Abstained" : "Selected value"}
          </p>
          <h3>
            {result.abstained ? "Not enough information" : (result.value ?? "No value")}
            {selectedProbability !== undefined && (
              <span className="result-percentage"> · {percent(selectedProbability)}</span>
            )}
          </h3>
          <dl className="result-meta">
            {alternative && (
              <div>
                <dt>{result.abstained ? "Top option" : "Runner-up option"}</dt>
                <dd>{alternative.option} · {percent(alternative.probability)}</dd>
              </div>
            )}
            {elapsedMs !== undefined && (
              <div>
                <dt>Completed in</dt>
                <dd>{(elapsedMs / 1000).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} s</dd>
              </div>
            )}
          </dl>
          <p className="muted">Model probabilities</p>
          <ul className="probabilities">
            {options.map((option, index) => (
              <li key={`${index}-${option}`}>
                <div>
                  <span>{option}</span>
                  <strong>{percent(result.probabilities[index])}</strong>
                </div>
                <meter
                  min={0}
                  max={1}
                  value={result.probabilities[index]}
                  aria-label={`Model probability for ${option}`}
                />
              </li>
            ))}
            <li className="abstention-probability">
              <div>
                <span>Not enough information</span>
                <strong>{percent(result.unknown_probability)}</strong>
              </div>
              <meter
                min={0}
                max={1}
                value={result.unknown_probability}
                aria-label="Model probability for not enough information"
              />
            </li>
          </ul>
        </div>
      )}
    </section>
  );
}
