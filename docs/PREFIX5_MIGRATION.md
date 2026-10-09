# Instruction-free prompt and five-token shared cache

The browser and Python use the same prompt body:

```text
<|im_start|>user
State: "JSON-quoted proposal text"
Question: question
A: first option
B: second option
C: unknown<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Only the generated instruction header is removed. State, question, option order,
unknown candidate, chat template, model weights, readout and calibration are retained.
The shared token IDs are `[248045, 846, 198, 1349, 25]`, ending at `State:`.
The following space/quote belongs to the suffix because tokenization can merge it
with an empty body or an escaped initial character. Original input text is retained.

The browser accepts at most 96 total tokens: five shared and 1–91 suffix tokens.
The browser builder imports the current paid API sources while retaining the
validated numerical kernels. Its non-chunked paid update path accepts five prefix
plus at most 89 suffix tokens (94 total); the browser query path independently
supports 96. Chunked paid builds accept up to 1024 total tokens. Past receipts
up to 1024 tokens remain replayable in either build, without expanding the
unchunked build's limit for new requests.
All 32 layer states occupy 3,816,448 bytes, versus 15,418,368 for the former cache.
Each manifest is content-addressed and binds its runtime module and prompt layout.
Execution plans are measured for the same module; stale caches and plans fail closed.
Measured plans use 32–63 queries. An opaque MLP generation step can cover up to
8,704 aligned rows, avoiding repeated carry transfers for long suffixes. The browser performs
protocol orchestration, with model arithmetic running in existing IC queries.

The instruction-free prompt was compared using 24 cases and 32 option orders:
22/24 and 28/32 correct, equal to the former format, without new regressions.
Current BOOM #584, #653 and #617 also passed. The five-token cache boundary leaves
those complete token sequences unchanged. Local inference evidence and diagnostic
tools remain under ignored artifacts, outside the public source tree.

This change stages a new release; deployment is a separate operation. Keep the old
Wasm, frontend build, release JSON and immutable prefix assets for recovery. Save
a canister snapshot before upgrading: the old runtime requires fee-version fields
that the current paid metadata no longer writes. To restore that runtime, restore
its snapshot together with its Wasm. Upgrade
the canister preserving its stable model, prepare its weight cache, publish the
verified browser prefix assets, verify the new module and query plans, then switch
the frontend. Do not deploy a
frontend pinned to the new module before the canister is ready.
