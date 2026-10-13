import { MAX_SUFFIX, MAX_TOKENS } from "./query-plan.ts";
import { useCallback, useEffect, useRef, useState } from "react";
import { ResultPanel } from "./ResultPanel";
import { TokenCounter } from "./TokenCounter";
import { samples, realWorldSamples } from "./samples";
import type { DecisionInput, DecisionResult, InferenceProgress } from "./types";
import type { TokenCounts } from "./tokenization.ts";
import { infer } from "./inference-client.ts";

function ledgerTokens(e8s: string) {
  const value = BigInt(e8s);
  return `${value / 100000000n}.${(value % 100000000n).toString().padStart(8, "0")}`;
}

type Option = { id: number; text: string };
const emptyInput: DecisionInput = {
  state: "",
  question: "",
  options: ["yes", "no"],
};

export default function App() {
  const [state, setState] = useState(emptyInput.state);
  const [question, setQuestion] = useState(emptyInput.question);
  const [options, setOptions] = useState<Option[]>([
    { id: 0, text: "yes" },
    { id: 1, text: "no" },
  ]);
  const nextId = useRef(2);
  const [questionTouched, setQuestionTouched] = useState(false);
  const [copyStatus, setCopyStatus] = useState("");
  const [copying, setCopying] = useState(false);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const activeSample = realWorldSamples.find((sample) => sample.id === selectedSample);
  const normalized = options.map((option) => option.text.trim());
  const optionErrors = normalized.map((text) =>
    !text
      ? "Enter an option."
      : text === "__unknown__"
        ? "This option is reserved for abstention."
        : normalized.filter((value) => value === text).length > 1
          ? "Options must be unique."
          : "",
  );
  const questionError = !question.trim() ? "Enter a question." : "";
  const valid = !questionError && optionErrors.every((error) => !error);
  const input = { state, question, options: options.map(option => option.text) };
  const inputKey = JSON.stringify(input);
  const [tokenAnswer, setTokenAnswer] = useState<{ key: string; counts: TokenCounts | null } | null>(null);
  const onCounts = useCallback((key: string, counts: TokenCounts | null) => setTokenAnswer({ key, counts }), []);
  const counts = tokenAnswer?.key === inputKey ? tokenAnswer.counts : null;
  const canRun = valid && counts?.paidPrefix === 5 && counts.paidSuffix !== null && counts.paidSuffix >= 1 && counts.paidSuffix <= MAX_SUFFIX;
  const [result, setResult] = useState<(DecisionResult & { elapsedMs: number }) | null>(null);
  const [resultOptions, setResultOptions] = useState<string[]>([]);
  const [resultAssessment, setResultAssessment] = useState(false);
  const [progress, setProgress] = useState<InferenceProgress | null>(null);
  const [inferenceError, setInferenceError] = useState("");
  const activeRun = useRef<ReturnType<typeof infer> | null>(null);
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; activeRun.current?.cancel(); }, []);
  function cancelInference() {
    generation.current++; activeRun.current?.cancel(); activeRun.current = null;
    setProgress(null); setInferenceError("Inference cancelled.");
  }
  async function runInference() {
    if (!canRun || activeRun.current) return;
    const id = ++generation.current, startedAt = Date.now();
    const startedTick = performance.now();
    const snapshot = structuredClone(input);
    setResult(null); setResultOptions(snapshot.options); setResultAssessment(activeSample?.id === "boom-617"); setInferenceError("");
    setProgress({ kind: "waiting", startedAt, phase: "Preparing" });
    const run = infer(snapshot, (completed, total) => {
      if (generation.current === id) setProgress({ kind: "steps", completed, total, startedAt, phase: "Running" });
    });
    activeRun.current = run;
    try {
      const answer = await run.promise;
      if (generation.current === id) setResult({ ...answer, elapsedMs: performance.now() - startedTick });
    }
    catch (error) { if (generation.current === id) setInferenceError(error instanceof Error ? error.message : "Inference failed."); }
    finally { if (generation.current === id) { activeRun.current = null; setProgress(null); } }
  }

  function changed() {
    setCopyStatus("");
    setSelectedSample(null);
    if (!activeRun.current) { setResult(null); setInferenceError(""); }
  }
  function loadSample(input: DecisionInput, id: string) {
    setState(input.state);
    setQuestion(input.question);
    setOptions(input.options.map((text) => ({ id: nextId.current++, text })));
    setQuestionTouched(false);
    changed();
    setSelectedSample(id);
  }
  async function copyInput() {
    setCopying(true);
    setCopyStatus("");
    try {
      await navigator.clipboard.writeText(
        JSON.stringify(
          {
            state,
            question,
            options: options.map((option) => option.text),
          } satisfies DecisionInput,
          null,
          2,
        ),
      );
      setCopyStatus("Input copied.");
    } catch {
      setCopyStatus(
        "Could not copy. Check your browser's clipboard permissions.",
      );
    } finally {
      setCopying(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="aurora" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <header className="site-header">
        <div className="site-header-inner">
          <a className="brand" href="#main" aria-label="KINIC — Imajev on IC">
            <img
              className="kinic-logo"
              src="/kinic-logo.svg"
              alt="KINIC"
              width="121"
              height="32"
            />
            <span className="brand-separator" aria-hidden="true" />
            <span className="product-identity">
              <span className="product-name">Imajev <span className="ic-label">on IC</span></span>
              <span className="model-label">Qwen 3.5-4B + adapter</span>
            </span>
          </a>
          <span className="connection-badge" title="Runs directly on the Internet Computer">
            <span aria-hidden="true" />
            <span className="connection-label">Public query</span>
          </span>
        </div>
      </header>
      <main id="main">
        <div className="intro">
          <h1>
            <span>From text </span>
            <span className="h1-accent">to decisions.</span>
          </h1>
        </div>
        <div className="workspace">
          <section className="input-panel" aria-labelledby="input-heading">
            <div className="panel-heading">
              <div>
                <h2 id="input-heading">Decision input</h2>
              </div>
            </div>
            <div className="sample-area">
              <p id="samples-label" className="sr-only">Start with an example</p>
              <div
                className="sample-buttons"
                role="group"
                aria-labelledby="samples-label"
              >
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    aria-pressed={selectedSample === sample.id}
                    title={sample.description}
                    onClick={() => loadSample(sample.input, sample.id)}
                  >
                    <span className="sample-title">
                      {sample.label}
                      <span className="sample-arrow" aria-hidden="true">
                        ↗
                      </span>
                    </span>
                  </button>
                ))}
              </div>
              <details className="real-world-examples">
                <summary>Real-world examples</summary>
                <div className="sample-buttons" role="group" aria-label="BOOM DAO proposals">
                  {realWorldSamples.map((sample) => (
                    <button
                      key={sample.id}
                      type="button"
                      aria-pressed={selectedSample === sample.id}
                      title={sample.description}
                      onClick={() => loadSample(sample.input, sample.id)}
                    >
                      <span className="sample-title">
                        {sample.label}
                        <span className="sample-arrow" aria-hidden="true">↗</span>
                      </span>
                    </button>
                  ))}
                </div>
              </details>
              {activeSample && (
                <div key={activeSample.id} className="sample-provenance">
                  <p className="sample-source">
                    Mechanically compacted input · Question added for this demo.
                    {" "}
                    <a href={activeSample.sourceUrl} target="_blank" rel="noopener noreferrer">
                      View {activeSample.sourceLabel} ↗
                    </a>
                  </p>
                  {activeSample.audit.ledger_snapshot && (
                    <div className="sample-ledger">
                      <p>Scenario: mint another 250M tokens using current balances. Proposal #653 already executed; this is a new hypothetical mint.</p>
                      <p>Ledger read: {activeSample.audit.ledger_snapshot.completedAt}. Model input uses enclosing ranges and a calculated share after the extra mint; exact ledger values are retained below.</p>
                      <details>
                        <summary>Ledger supply and recipient balance</summary>
                        <dl>
                          <dt>Total supply (tokens)</dt><dd>{ledgerTokens(activeSample.audit.ledger_snapshot.totalSupplyE8s)}</dd>
                          <dt>Recipient account balance (tokens)</dt><dd>{ledgerTokens(activeSample.audit.ledger_snapshot.recipientBalanceE8s)}</dd>
                          <dt>Calculated share after additional mint</dt><dd>{activeSample.audit.derived?.share_percent_bounds.join("–")}%</dd>
                          <dt>Calculation</dt><dd>(account balance + mint) / (total supply + mint)</dd>
                          <dt>Ledger</dt><dd>{activeSample.audit.ledger_snapshot.ledgerCanisterId}</dd>
                          <dt>Account owner (default subaccount)</dt><dd>{activeSample.audit.ledger_snapshot.recipient}</dd>
                        </dl>
                        <p>Read-only icrc1_total_supply and icrc1_balance_of queries, fetched separately between {activeSample.audit.ledger_snapshot.startedAt} and {activeSample.audit.ledger_snapshot.completedAt}. These values are a saved snapshot.</p>
                      </details>
                    </div>
                  )}
                  {activeSample.audit.participation_snapshot && (
                    <div className="sample-ledger">
                      <p>Scenario: apply #617 to current neurons without extending their locks. Experimental assessment of suspected malicious governance manipulation. This simulation does not establish intent or reconstruct the proposal-time outcome.</p>
                      <p>Eligible neurons: {activeSample.audit.participation_snapshot.analysis.before.count} → {activeSample.audit.participation_snapshot.analysis.after.count}. These are neurons, not a count of people.</p>
                      <details>
                        <summary>Voting participation snapshot and assumptions</summary>
                        <dl>
                          <dt>Neurons fetched</dt><dd>{activeSample.audit.participation_snapshot.totalNeurons}</dd>
                          <dt>Neurons losing eligibility</dt><dd>{activeSample.audit.participation_snapshot.analysis.excluded_neurons}</dd>
                          <dt>Share of previously eligible neurons losing eligibility</dt><dd>{activeSample.audit.participation_snapshot.analysis.excluded_percent_bps_floor === null ? "No eligible baseline" : `At least ${((activeSample.audit.participation_snapshot.analysis.excluded_percent_bps_floor ?? 0) / 100).toFixed(2)}%`}</dd>
                          <dt>Largest neuron’s share under old threshold</dt><dd>{activeSample.audit.participation_snapshot.analysis.before.largest_share_bps_floor === null ? "No eligible baseline" : `At least ${((activeSample.audit.participation_snapshot.analysis.before.largest_share_bps_floor ?? 0) / 100).toFixed(2)}% of indexed voting power`}</dd>
                          <dt>Largest remaining neuron’s share</dt><dd>{activeSample.audit.participation_snapshot.analysis.after.largest_share_bps_floor === null ? "No remaining voting power" : `At least ${((activeSample.audit.participation_snapshot.analysis.after.largest_share_bps_floor ?? 0) / 100).toFixed(2)}% of remaining indexed voting power`}</dd>
                          <dt>Fetched between</dt><dd>{activeSample.audit.participation_snapshot.startedAt} and {activeSample.audit.participation_snapshot.completedAt}</dd>
                          <dt>Snapshot hash</dt><dd>{activeSample.audit.participation_snapshot.sha256}</dd>
                        </dl>
                        <p>Eligibility requires positive indexed voting power and a dissolve delay at least as long as the threshold. Dissolving neurons use remaining time at the snapshot reference time. Concentration holds current indexed weights fixed; changed voting bonuses are not recalculated. The indexed API pages are fetched separately.</p>
                        <a href={activeSample.audit.participation_snapshot.sourceUrl} target="_blank" rel="noopener noreferrer">View neuron data ↗</a>
                      </details>
                    </div>
                  )}
                  <details className="sample-original">
                    <summary>Original payload and identifier mapping</summary>
                    <pre>{activeSample.originalState}</pre>
                    {activeSample.audit.identifiers.length > 0 && (
                      <dl>
                        {activeSample.audit.identifiers.map(identifier => (
                          <div key={identifier.alias}>
                            <dt>{identifier.alias}</dt>
                            <dd>{identifier.value}</dd>
                          </div>
                        ))}
                      </dl>
                    )}
                  </details>
                </div>
              )}
            </div>
            <form onSubmit={(event) => { event.preventDefault(); void runInference(); }} noValidate>
              <div className="field">
                <label htmlFor="state">
                  Context <span className="optional">Optional</span>
                </label>
                <textarea
                  id="state"
                  rows={3}
                  placeholder="Add context for your decision…"
                  value={state}
                  onChange={(event) => {
                    setState(event.target.value);
                    changed();
                  }}
                />
              </div>
              <div className="field">
                <label htmlFor="question">
                  Question <span className="required">Required</span>
                </label>
                <textarea
                  id="question"
                  rows={2}
                  placeholder="What would you like to decide?"
                  value={question}
                  required
                  aria-invalid={questionTouched && !!questionError}
                  aria-describedby={
                    questionTouched && questionError
                      ? "question-error"
                      : undefined
                  }
                  onBlur={() => setQuestionTouched(true)}
                  onChange={(event) => {
                    setQuestion(event.target.value);
                    changed();
                  }}
                />
                <p id="question-error" className="error" aria-live="polite">
                  {questionTouched && questionError}
                </p>
              </div>
              <fieldset className="choices" aria-describedby="abstention-help">
                <legend>
                  Options <span className="optional">2–7 options</span>
                </legend>
                <div className="option-list" data-compact={normalized.every(text => text.length <= 3)}>
                  {options.map((option, index) => (
                    <div className="option-row" key={option.id}>
                      <div className="option-control">
                        <label
                          htmlFor={`option-${option.id}`}
                          className="option-number"
                        >
                          {String(index + 1).padStart(2, "0")}
                          <span className="sr-only"> Option {index + 1}</span>
                        </label>
                        <input
                          id={`option-${option.id}`}
                          aria-label={`Option ${index + 1}`}
                          value={option.text}
                          aria-invalid={!!optionErrors[index]}
                          aria-describedby={`option-error-${option.id}`}
                          onChange={(event) => {
                            setOptions(
                              options.map((item) =>
                                item.id === option.id
                                  ? { ...item, text: event.target.value }
                                  : item,
                              ),
                            );
                            changed();
                          }}
                        />
                        <button
                          className="remove-button"
                          type="button"
                          aria-label={`Remove option ${index + 1}`}
                          disabled={options.length <= 2}
                          onClick={() => {
                            setOptions(
                              options.filter((item) => item.id !== option.id),
                            );
                            changed();
                          }}
                        >
                          ×
                        </button>
                      </div>
                      <p
                        className="error"
                        id={`option-error-${option.id}`}
                        aria-live="polite"
                      >
                        {optionErrors[index]}
                      </p>
                    </div>
                  ))}
                </div>
                <div className="option-tools">
                <button
                  className="add-button"
                  type="button"
                  disabled={options.length >= 7}
                  onClick={() => {
                    setOptions([
                      ...options,
                      { id: nextId.current++, text: "" },
                    ]);
                    changed();
                  }}
                >
                  <span aria-hidden="true">＋</span> Add option
                </button>
                <p id="abstention-help" className="abstention-help">
                  <span>Not enough information</span>{" "}
                  <span>→ Abstain</span>
                </p>
                </div>
              </fieldset>
              <div className="actions">
                <button
                  className="primary-button"
                  type="submit"
                  disabled={!canRun || progress !== null}
                  aria-describedby="connection-help"
                >
                  Run inference <span aria-hidden="true">→</span>
                </button>
                {progress && <button className="copy-button" type="button" onClick={cancelInference}>Cancel</button>}
                <button
                  className="copy-button"
                  type="button"
                  disabled={!valid || copying}
                  onClick={copyInput}
                >
                  Copy input
                </button>
              </div>
              <p id="connection-help" className="help">
                {progress ? "Running your submitted input." : `Up to ${MAX_TOKENS} tokens. No login required.`}
              </p>
              {inferenceError && <p className="error" role="alert">{inferenceError}</p>}
              <p
                className="copy-status"
                role="status"
                data-tone={
                  copyStatus.startsWith("Could not copy") ? "error" : undefined
                }
              >
                {copyStatus}
              </p>
            </form>
          </section>
          <div className="output-column">
            <ResultPanel
              result={result}
              assessment={resultAssessment}
              elapsedMs={result?.elapsedMs}
              progress={progress}
              options={result || progress ? resultOptions : input.options}
            />
            <TokenCounter
              input={input}
              onCounts={onCounts}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
