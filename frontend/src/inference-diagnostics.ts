/** Optional, per-run diagnostics. Durations may overlap; do not add them up. */
export type TimingPhase = "prefix-fetch" | "prefix-hash" | "prefix-wait" | "prefix-cache" |
  "agent-init" | "module-check" | "query" | "preparation" | "total";
export interface TimingEvent {
  phase: TimingPhase;
  durationMs: number;
  layer?: number;
  method?: string;
  bytes?: number;
}
export interface InferenceDiagnostics { events: TimingEvent[]; prefixCacheBytes?: number }
export type TimingObserver = (event: TimingEvent) => void;

export async function timed<T>(observer: TimingObserver | undefined, phase: TimingPhase,
  task: () => Promise<T>, detail: Omit<TimingEvent, "phase" | "durationMs"> = {}): Promise<T> {
  if (!observer) return task();
  const start = performance.now();
  try { return await task(); }
  finally { observer({ phase, durationMs: performance.now() - start, ...detail }); }
}
