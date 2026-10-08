import type { DecisionInput, DecisionResult } from "./types.ts";
import type { TokenCounts } from "./tokenization.ts";

let worker: Worker | undefined;
let nextId = 0;
type Message = { id: number; counts?: TokenCounts; completed?: number; total?: number; result?: DecisionResult; error?: string };
const pending = new Map<number, { resolve: (m: Message) => void; reject: (e: Error) => void; progress?: (n: number, total: number) => void }>();
function getWorker() {
  if (!worker) {
    worker = new Worker(new URL("./tokenizer.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<Message>) => {
      const message = event.data, entry = pending.get(message.id);
      if (!entry) return;
      if (message.completed !== undefined) { entry.progress?.(message.completed, message.total!); return; }
      pending.delete(message.id);
      if (message.error) entry.reject(new Error(message.error)); else entry.resolve(message);
    };
    worker.onerror = () => {
      for (const entry of pending.values()) entry.reject(new Error("Inference worker failed. Please retry."));
      pending.clear(); worker?.terminate(); worker = undefined;
    };
  }
  return worker;
}
export function countTokens(input: DecisionInput): Promise<TokenCounts> {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve: m => resolve(m.counts!), reject });
    getWorker().postMessage({ id, input, op: "count" });
  });
}
export function infer(input: DecisionInput, progress: (n: number, total: number) => void) {
  const id = ++nextId;
  const promise = new Promise<DecisionResult>((resolve, reject) => {
    pending.set(id, { resolve: m => resolve(m.result!), reject, progress });
    getWorker().postMessage({ id, input, op: "run" });
  });
  return { promise, cancel() {
    worker?.postMessage({ id, op: "cancel" });
    pending.get(id)?.reject(new DOMException("Inference cancelled.", "AbortError"));
    pending.delete(id);
  } };
}
