import { useEffect, useState, type CSSProperties } from "react";
import type { InferenceProgress } from "./types";

const clock = (ms: number) => {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [active]);
  return now;
}

/** Segments look crisp up to this many steps; beyond it a continuous bar is used. */
const MAX_SEGMENTS = 48;

/**
 * Shows only what the client actually observes: completed/total calls for
 * sequential queries, or elapsed time for a single opaque call. The remaining
 * time is an estimate from this run's own completed calls, never a constant.
 */
export function InferenceStatus({ progress }: { progress: InferenceProgress }) {
  const now = useNow(true);
  const elapsed = now - progress.startedAt;

  if (progress.kind === "waiting") {
    return (
      <div className="inference" data-kind="waiting">
        <p className="inference-phase">{progress.phase ?? "Running inference"}</p>
        <div
          className="inference-track is-indeterminate"
          role="progressbar"
          aria-label="Inference in progress"
          aria-valuetext={`Elapsed ${clock(elapsed)}`}
        >
          <span />
        </div>
        <dl className="inference-meta">
          <div>
            <dt>Elapsed</dt>
            <dd>{clock(elapsed)}</dd>
          </div>
        </dl>
        <p className="inference-note">
          Progress is unavailable during a single call.
        </p>
      </div>
    );
  }

  const total = Math.max(1, progress.total);
  const done = Math.min(progress.completed, total);
  const ratio = done / total;
  const remaining =
    done >= 3 && done < total
      ? (elapsed / done) * (total - done)
      : null;
  const segmented = total <= MAX_SEGMENTS;

  return (
    <div className="inference" data-kind="steps">
      <p className="inference-phase" aria-live="polite">
        {progress.phase ?? "Running inference"}
      </p>
      <p className="inference-count" aria-hidden="true">
        <span className="inference-done">{done}</span>
        <span className="inference-total">/ {total}</span>
        <span className="inference-unit">query</span>
      </p>
      <div
        className={`inference-track${segmented ? " is-segmented" : ""}`}
        role="progressbar"
        aria-label="Inference progress"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
        aria-valuetext={`${done} of ${total} queries completed`}
        style={{ "--ratio": ratio } as CSSProperties}
      >
        {segmented ? (
          Array.from({ length: total }, (_, i) => (
            <i
              key={i}
              className={
                i < done ? "is-done" : i === done ? "is-current" : undefined
              }
            />
          ))
        ) : (
          <span />
        )}
      </div>
      <dl className="inference-meta">
        <div>
          <dt>Elapsed</dt>
          <dd>{clock(elapsed)}</dd>
        </div>
        <div>
          <dt>Remaining</dt>
          <dd>{remaining === null ? "Estimating…" : `About ${clock(remaining)}`}</dd>
        </div>
      </dl>
    </div>
  );
}
