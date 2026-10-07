import { useCallback, useEffect, useRef, useState } from "react";
import { ResultPanel } from "./ResultPanel";
import { TokenCounter } from "./TokenCounter";
import { samples, realWorldSamples } from "./samples";
import type { DecisionInput, DecisionResult, InferenceProgress } from "./types";
import type { TokenCounts } from "./tokenization.ts";
import { infer } from "./inference-client.ts";

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
  const canRun = valid && counts?.paidPrefix === 27 && counts.paidSuffix !== null && counts.paidSuffix >= 1 && counts.paidSuffix <= 57;
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [resultOptions, setResultOptions] = useState<string[]>([]);
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
    const snapshot = structuredClone(input);
    setResult(null); setResultOptions(snapshot.options); setInferenceError("");
    setProgress({ kind: "steps", completed: 0, total: 32, startedAt, phase: "Preparing" });
    const run = infer(snapshot, completed => {
      if (generation.current === id) setProgress({ kind: "steps", completed, total: 32, startedAt, phase: "Running" });
    });
    activeRun.current = run;
    try { const answer = await run.promise; if (generation.current === id) setResult(answer); }
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
                <p className="sample-source">
                  Original payload excerpt · Question added for this demo.
                  {" "}
                  <a href={activeSample.sourceUrl} target="_blank" rel="noopener noreferrer">
                    View {activeSample.sourceLabel} ↗
                  </a>
                </p>
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
                <div className="option-list">
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
                {progress ? "Running your submitted input." : "Up to 84 tokens. No login required."}
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
