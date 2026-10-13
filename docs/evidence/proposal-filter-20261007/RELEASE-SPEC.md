# imajev-4b — release specification (phase-3 version, staged 2026-09-26)

This folder is the version that ships to Hugging Face (`mohit67890/imajev-4b`), the Space and the public repo. The previous
version (phase-2c soup50, released 2026-09-24) is kept in `../_previous/phase-2c/imajev-4b/`; raw phase-3 outputs are in
`imajev/reports/phase3/train-results/` (every checkpoint's panels, benchmarks, gates and the DecisionBench tracking files).

## What the adapter is
- Base: `Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (frozen).
- Adapter: LoRA **rank 64 / α 128** (scale 2, unchanged) on the language layers incl. the DeltaNet projections + trained **256-code**
  decision readout (255 option codes + unknown; questions with ≤ 254 options use the same codes as the previous 255-code readout),
  PEFT format at the root, `mlx/` = the same weights converted with `scripts/convert_peft_adapter_to_mlx.py` (400 tensors, 0 unmatched,
  readout copied).
- Recipe (phase 3, `imajev/docs/phase-3-plan.md`, run log `imajev/docs/handoff-phase3-2026-09-26.md`): start from the released phase-2c
  soup50 adapter with its rank-16 LoRA expanded to rank 64 (output identical at step 0) and the readout extended to 256 codes; **round 1**:
  2 epochs over `decision-p3` (125,424 rows: 46% hard text, 16% earlier text, 26% hard image incl. 20,270 constructed chart / document /
  inventory / safety / geometry / screen decisions, 12% earlier photos; failures of the released model mined from a 210k pool, labelled by
  Qwen3.6-35B-A3B (thinking) and reviewed), lr 2e-5, soft targets, option permutation, rationale loss 0.3, 2,640 steps on 8×H100 (3 h 13 min);
  **round 2**: the 13,247 items the round-1 model still failed + 13,247 replay, 1 epoch, lr 1e-5, 291 steps (23 min). Pick = the last round-2
  snapshot `r2-s000291` (pilot ablations chose rank 64 and 256 codes; the ordinal loss lost).
- Why this checkpoint and not a soup: it passes the image gate on its own (ImajevBench 83.9, joint 100/122 vs the 97 bound), the state, pairs,
  irrelevance and false-abstention gates, and is the best by held-out score. It **fails one phase-2c gate**: 11 of 14 unknown-gold rows on the
  hard-question test are answered instead of abstained (bound 14/14, zero tolerance; misses at 0.83 / 0.60 / 0.51 confidence; false abstention
  0.5% on the 421 answerable rows). The owner overrode that gate and stopped the soup fallback; this is disclosed on the model card.

## Calibration (`calibration.json`, `calibration-rot4.json`; schema 1.1)
- Both files hold the same single temperature fitted by NLL on the 150-item authored JevBench-style dev set (the per-type fit on the flagged
  held-out half was rejected by the pooled-ECE guard: 0.059 single / 0.039 rot4 > 0.03). Unknown offsets 0.

## Serving configuration (what the numbers below were measured with)
- Direct option scoring, **4 cyclic rotations averaged** (`--rotations 4`), calibration applied, single pass, no reasoning tokens.
  `scripts/playground/server.py` (`--backend mlx` on a Mac, `--backend torch` on CUDA); the readout size and LoRA rank are read from the files.

## Benchmarks (pod ctr4dy9bzom5ji, 8×H100, torch evaluator, 2026-09-26; shipped 1.0 re-measured on the same pod)
| Benchmark | This version (r2-s000291) | Previous (phase-2c soup50, same pod) |
|---|---|---|
| JevBench public hard (111) | raw 71.2 / ECE 0.113 · cal 71.2 / 0.082 · rot4 72.1 / 0.079 · **rot4+cal 72.1 / 0.082**; pooled public ECE (cal) 0.046 | raw 69.4 / 0.164 · cal 69.4 / 0.109 · rot4+cal 70.3 / 0.116; pooled 0.038 |
| JevBench original / easy | 98.6 / 100 | 98.6 / 100 |
| ImajevBench v2.0-lite public test (279, direct scoring, full rotations) | **83.9** (joint 100/122, text 25/37, visual 109/120) | 82.4 (97 / 26 / 107) |
| ImajevBench private-1 hidden (202) | **85.6** (173/202; joint 70/88, text 20/30, visual 83/84; ECE 0.029; Mac MLX run) | 84.2 |
| Own held-out: fresh-seed 4,297 / human-verified 785 / flagged 2,813 | **87.2 / 76.2 / 79.0** | 68.4 / 48.2 / 23.9 |
| Constructed image sets (300/300/250/250/250/250): charts / docimg / inventory / safety / geometry / screens | 95.7 / 90.3 / 67.6 / 92.4 / 88.4 / 95.6 | 84.0 / 76.0 / 54.8 / 72.0 / 66.4 / 72.8 |
| DecisionBench 3k stratified subset (tracking only; single pass raw): acc / full-suite equiv / ordinal / reasoning | 72.9 / 78.5 / 61.3 / 80.4 | 72.2 / 78.2 / 49.1 / 67.1 |
| **DecisionBench 1.0 full suite, official harness** (8×H100 pod, 2026-09-26, 64k token limit): primary / scored-row / coverage / ECE / reasoning / ordinal | **79.69 / 79.69 / 100% (0 errors, 0 unsupported) / 0.069 / 80.6 / 46.2** | 77.55 / 79.33 / 97.75% (537 unsupported) / 0.024 / 68.0 / 40.3 |
| fast-decisions dev (1,700 rows / 2,900 heads): domain macro / pooled heads | 60.4 / 59.4 | 59.0 / 58.4 |
| Ship gates | state probe 76.5, pairs 96.7, irrelevance 80.0, false abstention 0.48%, image gate PASS; **unknown-gold 11/14 (FAIL, overridden)** | all PASS (unknown 14/14) |
| Typed S1-Bench (our conversion, 212 EN items; not an S1-Bench score) | 99.1 (ECE 0.019) | 98.6 (0.006) |

As served on a Mac (MLX `mlx/` weights, 4 rotations + `calibration-rot4.json`, `scripts/playground/server.py --backend mlx`, 2026-09-26 16:21-16:24 IST):
JevBench hard **71.2 / ECE 0.073**, original 98.6 / 0.069, easy 100 / 0.006 (torch on the pod: 72.1 / 0.082; one hard item apart, the same
MLX-vs-torch spread as the previous release). Files: `imajev/reports/phase3/release-check/jevbench-rot4cal/`.
MLX vs torch parity on ImajevBench public test: 234/279 both, argmax agreement 275/279 (98.6%), MLX ECE 0.059 (torch 0.070) —
`imajev/reports/phase3/release-check/public-test/score`.
All 8 checkpoints side by side: `imajev/reports/phase3/train-results/benchmarks.md`. Raw files: `.../p3/run/eval/r2-s000291/`,
`.../p3/tracking-only/r2-s000291/decisionbench.json`.

## File hashes (sha256) — see `SHA256SUMS` in this folder (PEFT files, calibration files and the `mlx/` export).

## Status
- Staged 2026-09-26 16:20 IST. MLX export, MLX serving check (JevBench + ImajevBench parity) and the hidden-split run all done (above). Ready to upload.
