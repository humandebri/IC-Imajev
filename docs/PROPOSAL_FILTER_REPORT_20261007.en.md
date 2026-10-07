# Imajev: evaluation as a filter for dangerous proposals

[日本語](PROPOSAL_FILTER_REPORT_20261007.md)

Evaluation date: October 7, 2026  
Scope: BOOM DAO proposals #500–#660, 161 proposals  
Evidence: frozen snapshots retrieved on October 6, 2026

## Purpose and conclusion

We evaluated Imajev as a filter intended to prevent dangerous proposals from reaching subsequent processing. This report maps the existing labels to a proposed operating policy: approve passes; reject and hold stop. Integration with downstream processing, automatic blocking, and tests of that integration were not performed. Hold includes uncertainty, insufficient evidence, and execution budget overflow; it does not establish that a proposal is dangerous.

Among the 50 proposals evaluated by the model, an agent review identified 32 as substantially tightening numerical participation requirements. The final labels were reject for 10 and hold for 22, so all 32 would stop under this policy. Holding a proposal that warrants rejection still serves this filtering purpose. The priority is whether dangerous content passes as approve, rather than agreement between three labels.

Blocking both reject and hold would prevent these 32 reviewed proposals from passing. However, the classification is a conditional agent review, not ground truth established by human reviewers. The results do not establish a general danger detection rate or the safety of entire proposals.

## What Imajev is

### A decision model that receives evidence and choices

Imajev is adapted to read evidence such as text, records, and photographs, and answer using choices defined by the user. Intended uses include identifying inconsistencies between product photographs and descriptions, routing inquiries to a department, and deciding whether a proposed change meets stated criteria. These general uses are separate from this proposal filter evaluation.

Upstream requests combine `state`, containing the evidence, with `questions`, defining questions, choices, and criteria. Images can also be supplied. Responses include typed choices or numerical judgments. Standard upstream outputs include probabilities for individual choices, `unknown_probability`, and `abstained`. Unknown is learned as a choice for cases where the evidence does not support selecting an answer.

This repository ports the **text decision path** of a pinned Imajev-4B release to the IC. This evaluation used numerical state, one question, and approve/reject choices, without an image encoder, image positions, or generative decoding. It did not validate every upstream question format or image feature on the IC.

### The roles of Qwen, LoRA, and the decision readout

Qwen3.5-4B is the text processing backbone. Imajev-4B adds LoRA for decision tasks and a dedicated readout that scores choices directly. LoRA adds the product of smaller A/B matrices to each projection. Here its rank is 64 and its scale is alpha/rank = 128/64 = 2. The large backbone weights and the additional decision weights can therefore be handled separately.

After the input passes through 32 layers, final RMSNorm normalizes the hidden vector at the last decision position. The dedicated readout is an F32 matrix of shape 256 × 2,560, representing 255 choice codes and unknown. Each question's choices are assigned codes, and the corresponding rows produce scores. The model does not generate an approve/reject sentence for a separate classifier to interpret.

This path evaluates the required A/B/unknown readout rows instead of a full vocabulary language generation head. It avoids unnecessary vocabulary rows and repeated generation, but still requires input embeddings and all 32 layers. A dedicated readout does not replace the preceding 4B model computation with a small classifier.

### The trained release used here

The pinned artifact is the final phase-3 checkpoint, `r2-s000291`. According to its release specification, LoRA and the readout were trained with the backbone frozen, using text and image decision examples, added difficult examples, choice order permutations, and teacher distributions. The model was not additionally trained or fine-tuned on these 161 BOOM DAO proposals. Proposal routing policy and numerical compression are defined in this repository's harness.

Upstream standard evaluation includes a configuration that averages four rotations of choice order and applies calibration. This evaluation used one inference with a fixed order and a threshold on the raw A/B logit difference. Upstream benchmark values and probabilities therefore cannot be transferred directly to this filter. The learned unknown choice and the harness's hold label are also separate mechanisms.

The architecture and trained release descriptions are based on the pinned [release specification](evidence/proposal-filter-20261007/RELEASE-SPEC.md) and [model card](evidence/proposal-filter-20261007/MODEL_CARD.md). They describe the retrieved revision, not a survey of the latest upstream state.

## Results for all 161 proposals

|Result|Count|Treatment as a filter|
|---|---:|---|
|approve|2|Pass the numerical participation filter|
|reject|10|Stop|
|hold|61|Stop; request further review where needed|
|Out of scope / excluded|88|Untested; not counted as passing|
|Total|161|A processing route was assigned to every proposal|

Of the 73 selected proposals, 2 passed and 71 stopped. The model evaluated 50; the other 23 received hold from tool checks for scope, evidence, or input budget.

The 88 exclusions include Motion proposals and display or branding changes excluded by the current selection policy. Exclusion does not establish safety. A downstream integration should treat these as an untested route.

## Would the filter stop dangerous participation changes?

The 50 model-evaluated proposals were checked against the originals and reviewed under the existing policy of limiting participation barriers.

The 32 below are a subset of those 50, not the total number of dangerous proposals among all 161. For example, #585 also contains substantial participation restrictions, but is excluded from this count because the tool held it for budget overflow. The 88 excluded proposals were not assessed for danger.

|Conditional review classification|Count|Actual final labels|Filter assessment|
|---|---:|---|---|
|Substantial increases in stake or minimum voting lock restrictions|32|10 reject, 22 hold|All 32 would stop; none passes as approve|
|Minimum voting lock changes from 1 to 2 days|16|16 hold|Stop; insufficient basis to conclude harm|
|Numerical participation requirements improve|2|2 approve|Pass; broader proposal risks require separate review|

For example, #562 and #566 increase minimum stake from 5 to 1 million tokens and minimum voting lock from 2 to 1,461 days. Both old and new values remained in the input, but the model's A/B output weakly favored approve. Its uncalibrated score was below 0.6, so the final label was hold and the proposals would not pass.

#587 and #588 also favored approve in the raw A/B output despite substantial restrictions, but ended as hold. These four cases show why the filter must use final results that include threshold and evidence checks, rather than the model's binary direction alone.

Sixteen proposals, including #602, increase minimum voting lock from 1 to 2 days. This is stricter, but no criterion establishes that this change alone is dangerously restrictive. Holding them for further review fits the filter's purpose. They cannot, however, be counted as safe proposals incorrectly stopped.

## Scope of the two passing decisions

|ID|Numerical change|Basis for passing and remaining questions|
|---|---|---|
|656|Minimum stake: 15 million → 5 tokens; voting period: 1 → 3 days|Participation requirements improve. The original summary's reference to a takeover is absent from the numerical input; intent and control remain unassessed|
|659|Minimum stake: 15 million → 5 tokens; minimum voting lock: 2 → 1 day; lock bonus: 1 → 100%|Eligibility requirements improve. The bonus change's effect on voting power distribution remains unassessed|

Approve means only that the proposal passes this numerical participation check. It does not establish safety across funds, control, executable code, or other aspects of the proposal, and does not authorize automatic voting.

## Breakdown of the 61 holds

|Reason for stopping|Count|Meaning|
|---|---:|---|
|Model A/B score below 0.6|38|The binary output difference is too small|
|Outside direct numerical stake / minimum voting lock scope|17|The current model policy cannot fully assess the content|
|Missing or conflicting evidence about funds, minting, or execution|5|Required evidence is unavailable|
|Input retaining all changes exceeds 128 tokens|1|#585 requires 130 tokens; held without truncation|

The five evidence cases comprise three treasury transfers, one mint, and one generic call. Checks against original data confirmed missing evidence about recipient control, balances, or supply, and a discrepancy between a generic call's structured payload and rendering. These findings support hold.

## Input harness and execution verification

All 161 routes were recalculated with the locally imported proposal_assessment numerical participation harness. We applied exact unit conversion from compact_units.py and the compression, binary decision, and evidence checking flow used by benchmark_600_660_binary.py.

The 50 model proposals were grouped into 18 identical numerical inputs and evaluated with fresh inference. Previous predictions and prefix states were not reused. Identical numerical input does not imply identical purpose or intent, so this deduplication is limited to numerical participation decisions.

The 128-token limit includes the question, choices, and chat special tokens. Old and new values and units for every changed field were retained. Evidence was not split into fragments for separate decisions, and inputs were not uniformly shortened to 86 tokens.

For all 50 model proposals, we verified snapshot hashes, agreement between specified structured payload values and the New rendering, and agreement between the old/new differences and the task's complete set of changed fields. Old values came from historical rendering and were not independently checked against the chain state at that time. Numerical preservation does not establish preservation of intent, purpose, or unknown real-world effects.

For all 18 inputs, we checked execution of 32 layers, model/input/module hashes, query counts, absence of replay or fallback, and resource bounds. All 18 logits matched the previous evaluation using the same numerical harness bit for bit. This establishes execution reproducibility, not semantic correctness.

|Execution item|Result|
|---|---:|
|Fresh distinct model inputs|18|
|Model inference queries|774|
|Fresh prefix and codec preparation queries|990|

Module verification and similar checks were counted separately. Model weights and canisters were not updated, and no votes were executed. Subsequent evidence review and threshold comparisons used no additional inference and did not change the original evaluation results.

## Execution environment and Imajev configuration

### Local execution environment

The evaluation ran on a local Internet Computer network. Proposals originated from a public SNS API, but inference read retained snapshots and queried localhost canisters. This was not mainnet inference, voting, or a mainnet cost measurement.

|Item|Configuration|
|---|---|
|Repository|`IC-Imajev`; file paths are relative to the repository|
|Host OS and architecture|macOS 27.0.1, build 26A434; arm64|
|Host tools|Python 3.12.14; icp-cli 1.0.2|
|Local endpoint|`http://localhost:8001/`|
|Inference canister|`7st3i-3l777-77775-aaaja-cai`|
|Prefix codec canister|`7vs54-wt777-77775-aaajq-cai`|
|Execution Wasm|`artifacts/merged-query32-v1/build/full.wasm`|
|Communication bridge|`artifacts/query32-v1/client-build/imajev-client`|
|Driver|`artifacts/proposal-query-optimization-v1/final-runner.py`|

OS, Python, and icp-cli versions were checked when the report was expanded, not recorded at inference startup. CPU model and host RAM were not collected for this evaluation. Execution-time module hashes establish canister and codec identity:

- Inference module: `6ce099e1f128b835b153bb28cde9f27dee719711be3822a38976cb4688613de4`
- Codec module: `e9644266f9bea8efdff5c1d5017b7c82beb67c3030279e6f27248b90fa51ef6e`

### Model and numerical arithmetic

The repository's Rust/Wasm runtime executes Qwen3.5-4B with the Imajev-4B LoRA adapter and dedicated decision readout. This is a text decision path that processes all 32 layers to extract choice scores, rather than generating free-form text.

|Item|Pinned configuration|
|---|---|
|Backbone|`Qwen/Qwen3.5-4B`|
|Backbone revision|`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`|
|Adapter|`mohit67890/imajev-4b`|
|Adapter revision|`c9e5f132465da85d31735ec502d5557982671a7d`|
|LoRA|Rank 64; alpha 128|
|Text decoder|32 layers; hidden size 2,560; MLP intermediate size 9,216|
|Layer composition|24 linear attention / Gated DeltaNet layers; 8 full attention layers|
|Quantized pack size|4,702,451,200 bytes, approximately 4.70 GB|
|Arithmetic|`int8-block256-base-f32-lora-v1`|
|Wire codec|`bf16-block256-exact-v1`|

Dense backbone weights use row-wise INT8. Backbone projection activations use per-token block256 INT8. LoRA and the readout use separate F32 arithmetic, retaining required F32 recurrent state and other values. Lossless wire compression and internal INT8 quantization are distinct operations; the arithmetic is not identical to the original BF16 model. Total pack size is also distinct from query heap usage.

The pinned model identity hash is `7ef38ecb70f4afa4e92bad2fe9f0448699ee45041688b1110819143f41601330`, and the pack hash is `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`. MODEL_LOCK.json records backbone, adapter, and tokenizer file identities.

Python handles snapshot extraction, exact unit conversion, tokenization, execution order, intermediate state storage, and decision gates. Numerical layer computation runs in ordinary queries. The client retains required state and passes it to the next call. Layer and operation fusion, including terminal readout integration, reduce query counts; 32 queries do not represent 32 independent decisions.

The readout computes three logits: A, B, and internal unknown. The binary harness uses only raw A/B, mapping A to approve and B to reject, and applying the threshold to `score = sigmoid(abs(logit_A - logit_B))`. Hold comes from threshold, evidence, scope, and budget checks, not directly from internal unknown. Upstream calibration settings do not make this A/B score a calibrated probability of correct danger detection.

### Input tokens and prefixes

Input includes old/new numbers, the question, A/B choices, chat template special tokens, and the decision position. It is tokenized with the pinned tokenizer and chat template, with thinking disabled. The 128-token limit is the proposal harness budget, not the backbone's context limit. Driver settings such as `token_cap=132` and `mlp_full_token_cap=89` serve different internal purposes and are separate from the allowed input limit of 128.

A prefix is a matching sequence at the beginning of an input. It is computed through all layers in advance, retaining hidden vectors, attention K/V, DeltaNet state and logs, and other required data. Suffix execution resumes from that state. A prefix is not a summary or deletion of input tokens; it substitutes already computed state for the same initial sequence.

These prefixes include shared chat content and matching initial numerical state. Eleven banks were prepared afresh and shared only between inputs with identical initial token sequences. Token, model, pack, module, and state file hashes are checked, preventing direct reuse with a different prefix or weights. Previous prefix states and predictions were not reused; the pinned model pack and other fixed artifacts already existed.

|Token measure|Value|
|---|---:|
|Input length across 18 distinct inputs|67–128 tokens|
|Total input tokens across 18 distinct inputs|1,861|
|Total input tokens counting all 50 proposals individually|4,799|
|Lengths of 11 prefix banks|8–48 tokens|
|Suffix lengths on the measured 32-query path|58–59 tokens|
|Suffix lengths on the measured 50-query path|77–80 tokens|

These totals describe complete logical input lengths, not prefix preparation work, inter-query state transfer, layer computation, or billable API tokens. Output uses a fixed-choice readout; generated text output tokens were not counted.

### The 18 distinct inputs and execution paths

Prefix length plus suffix length equals total input length. Rows containing multiple proposal IDs share one fresh inference because their numerical inputs are identical.

|Proposal ID|Total tokens|Prefix bank|Prefix tokens|Suffix tokens|Inference queries|
|---|---:|---:|---:|---:|---:|
|505|128|0|48|80|50|
|552|112|3|32|80|50|
|553|67|9|9|58|32|
|554|68|9|9|59|32|
|555, 556|68|9|9|59|32|
|559, 563, 567|88|7|29|59|32|
|562, 566|123|1|46|77|50|
|570, 571, 574, 577, 580|110|4|30|80|50|
|586|126|1|46|80|50|
|587|126|1|46|80|50|
|588|126|1|46|80|50|
|590, 591, 592, 593|126|2|46|80|50|
|594, 595, 596, 597, 601|124|2|46|78|50|
|598, 599, 600|126|2|46|80|50|
|602, 605, 608, 611, 614, 620, 623, 626, 629, 632, 635, 638, 641, 644, 647, 650|67|10|8|59|32|
|617|97|6|38|59|32|
|656|78|8|19|59|32|
|659|101|5|21|80|50|

The earlier fixed prefix of 27 tokens plus a suffix of 59 gave a total of 86. Neither 86 nor 87 is a universal Imajev input limit. For this module, the driver admits a 32-query path with prefixes of 1–38 tokens and suffixes of 1–59; not every combination was measured. #617 used 38 + 59 = 97 tokens in 32 queries. The harness also retains longer inputs up to 128, without guaranteeing 32 queries. #505 used 48 + 80 = 128 tokens in 50 queries.

### Prefix preparation and measured cost

Each bank required 66 queries to compute the model prefix and 24 queries to prepare states for the 24 linear attention layers with the codec: 90 per bank, or 990 for 11 banks. All prefix and codec preparation was fresh; those states were then shared between inputs using the same prefix.

|Category|Queries|Sum of recorded processing time|
|---|---:|---:|
|Suffix inference for 18 distinct inputs|7 × 32 + 11 × 50 = 774|695.64 seconds|
|Model prefix preparation for 11 banks|11 × 66 = 726|202.09 seconds|
|Codec preparation for 11 banks|11 × 24 = 264|198.18 seconds|
|Total above|1,764|1,095.91 seconds, approximately 18.27 minutes|

Time is the sum of processing durations stored in reports, not elapsed time from task start to completion. It excludes some costs, such as process startup and module checks. Recorded suffix inference time per input ranged from 22.20 to 74.95 seconds. These local measurements do not establish mainnet latency or throughput.

The 32/50-query figures count inference calls for one input after prefix preparation. They exclude initial prefix and codec preparation, module/state-tree checks, model weight upload, and fixed preparation updates. The 1,764-query total is not the complete cold-start cost from model installation. Reporting only 32 queries as the initial total cost would omit required preparation.

|Suffix inference verification measure|Maximum observed|
|---|---:|
|Handler instructions per query|4,924,884,361; checked against a strict bound of 5 billion|
|Query heap|4,140,892,160 bytes; checked against a strict bound of 4 GiB|
|Request per query|1,940,728 bytes; checked against a strict bound of 2 million bytes|
|Reply per query|1,708,182 bytes; checked against a strict bound of 2 million bytes|

Instruction counts are handler measurements excluding CDK Candid decoding and encoding. Request/reply sizes describe recorded Candid data, excluding HTTP, CBOR, signatures, and other overhead. Heap is observed linear memory at query completion, not allocator usage or a strict instantaneous peak during execution. The observed bounds were met, but maxima are close to the limits, so these results cannot be extended unconditionally to untested lengths or paths.

## Optimizations for execution on the IC

### Partitioning computation to fit resource limits

This experiment had to satisfy query instruction budgets, request/reply size limits, and Wasm32 heap capacity simultaneously. The entire model could not fit into one query. Excessively fine partitioning repeats state transfer, encoding, decoding, and verification; simply joining queries can exceed instruction or 2 MB communication budgets. Optimization therefore reduced computation and adjusted partition boundaries to fit the limits.

A query here computes part of the model. One decision on the full input requires multiple queries. Weight installation and fixed preparation use updates; numerical model computation for these inputs used ordinary queries. The client carries per-question intermediate state between queries instead of persisting it between query calls on the canister.

### Weight quantization and fixed preparation

Only weights required for the text path were exported, excluding vision, MTP, and the ordinary generation head. The initial unquantized text pack was 8,901,719,552 bytes; the INT8 pack is 4,702,451,200 bytes, a reduction of approximately 47.17%. Backbone matrices and embeddings use INT8, while LoRA and the dedicated readout remain F32 and required scalars and norms retain BF16/F32. Conversion conditions are recorded in [initial full-layer integration and packing](FULL_INFERENCE.md).

Prepared caching reduces repeated stable memory reads, unpacking, and copying of fixed INT8 weights. F32 LoRA and readout weights are also restored and checked for finite values during preparation; queries borrow immutable slices. The fixed cache contains 272 INT8 dense tensors and 449 F32 tensors: 721 tensors totaling 4,065,416,192 bytes. Required embedding rows and some small weights are still read from the stable pack.

Weight layout conversion, RoPE sin/cos, and BF16 activation tables also move to fixed preparation. Activation tables store output bits from the original Wasm arithmetic rather than introducing a different approximation. This trades preparation work and heap capacity for less repeated query work. The 721 weight preparation updates are excluded from the 774 inference queries and 990 prefix preparation queries. See [prepared F32 weights](PREPARED_F32.md) and [activation tables](PREPARED_ACTIVATION.md).

### INT8 integer dot products and SIMD kernels

Initially, INT8 weights were expanded to F32 for multiply-accumulate operations. The later path quantizes backbone projection inputs into per-token blocks of 256 columns, computes dot products in I32, and applies activation and weight scales. LoRA receives the original F32 input; recurrent F32 state and the readout are not all quantized to INT8.

Historical measurements for the same 132-token input reduced handler instructions from 1,761,784,928,449 to 1,022,432,919,278, approximately 41.97%. Input quantization and accumulation order changed, so this does not claim bit identity with the original F32 path. Numerical identity of the current 32/50-query implementation is measured against the adopted integer arithmetic. See [integer arithmetic and numerical differences](INTEGER_ARITHMETIC.md).

Within that arithmetic, Wasm SIMD shares weight and input loads across outputs and tokens. Inner loops use constant expansion, fewer temporary arrays, and connected dot/scale operations. Multi-token INT8 projections also use shallow Strassen-style transforms while preserving integer computation within 256-column blocks and the F32 addition order. The terminal single-token path uses direct SIMD instead of zero-padding one side of a two-token kernel.

F32 LoRA shares input loads across outputs. DeltaNet retains its 128 × 128 state inside the kernel to reduce repeated loads, stores, and decay calculation. The frozen Wasm used here substitutes six WAT kernels for INT8 projections, single-token projections, F32 LoRA, Delta state retention, and related operations, and passed wasmparser validation. Identity is based on the retained build report and module hash, not an assumption that rebuilding current sources produces the same Wasm.

### Sharing input quantization and LoRA A products

Q/K/V and MLP gate/up perform different projections on the same input. Results are shared within a query to avoid repeating block256 quantization and the same LoRA A products. When output row partitioning crosses query boundaries, the quantized input, scales, and original F32 A products travel as client-held carry for direct use by the next query.

Carry is an intermediate computational value, not a cached final decision. Numerical values and progress are retained, with shape, model, pack, input, and step checks. This enables reuse of identical work while preventing reuse with the wrong input.

### Lossless communication and prefix state compression

Sending every intermediate value as four-byte F32 increases traffic and message size. `bf16-block256-exact-v1` stores exactly BF16-representable blocks using two bytes per element, and retains four bytes where F32 is required. It introduces no additional rounding and preserves recurrent F32 state bits. Residuals and partial sums also use lossless plane and dictionary compression to fit required values into communication budgets.

Sending large F32 DeltaNet states for every head consumes substantial space. The implementation combines exact reconstruction from original update logs with a hybrid path that transmits some compressed state. Early standalone codec documents describe it as not yet integrated with full inference. For this evaluation, commands, cache identities, and full reports confirm hybrid packets connected to the prefix-start path.

The 11 banks precompute longer matching sequences than shared instructions alone, shortening suffix computation without changing inputs. A new input that does not match a bank needs additional preparation. Preparation cannot be shared for free across arbitrary proposals.

### Operation fusion and query scheduling across layers

Queries connect residual addition and RMSNorm; MLP gate/up → SwiGLU → down; attention K/V → Q → norm/RoPE → attention → gate; and Delta projection → convolution → recurrence → output projection. This reduces intermediate replies and resubmission. Already validated values also pass directly between operations within a query, avoiding repeated encoding, decoding, and copying. External input validation is retained.

Prefix-start connects suffix token IDs to embeddings, the first norm, and Delta computation. Later roll/join/tail paths place parts of the next Delta or attention operation into a query completing an MLP. The 50-query path uses this scheduling across layers.

The 32-query path follows the repeating group of three linear attention layers and one full attention layer. It joins the remaining MLP, next attention, and beginning of the next MLP, scheduling four layers in four queries. Output rows and heads per query are balanced to avoid a single query exceeding the instruction budget. The initial allocation for #617 exceeded the limit, so the validated balanced v3 allocation was used.

Some intermediate fusion versions enlarged carry and increased communication or total instructions. Fusion alone does not establish a speed improvement. Query counts, instructions, communication, and heap were measured separately.

### Omitting unnecessary terminal computation and replies

The decision needs the hidden vector at the final position. Unnecessary Q/output computation in the last attention operation and full-token final MLP work are omitted. Earlier hidden values and K/V needed by the final token are still computed. This does not skip reading the beginning of the input.

The final two layers' MLP, attention, norm, and readout connect as a terminal tail. States that will not continue and hidden values consumed internally are not returned. Optimization comparisons check the 31 retained layers of hidden values, 32 layers of conv/KV/position, final normalized hidden, logits, and other retained values. They do not claim a direct comparison of the unreturned layer30 hidden.

### Effects measured without changing the inputs

The retained original execution and this fresh execution use the same 18 inputs. Questions, choices, numbers, model pack, and threshold are fixed, separating these reductions from those obtained by changing input representation.

|Measure|Retained original execution|Fresh execution|
|---|---:|---:|
|Inference queries across 18 inputs|3,703|774|
|Fresh prefix and codec preparation queries|0|990|
|Preparation plus successful inference queries|3,703|1,764|
|Raw logits and final labels per input|Comparison baseline|All 18 match|

Inference queries fell by approximately 79.10%. Successful execution queries including this preparation fell by approximately 52.36%. This comparison covers the 18 fixed inputs and excludes fixed weight preparation, upload, and module checks. It is not a reduction guaranteed for any single arbitrary input.

Earlier scheduling exploration also involved 15 successful queries, 3 queries exceeding limits, and 32 additional entry validation queries. That exploration was recorded separately as 774 + 990 + 18 = 1,782 queries, or 1,814 including entry validation. Its cost is not added to this fresh 1,764-query rerun, and exploration is not described as free.

Preserving logits and decisions does not improve semantic decision accuracy. Cases where the model favors the wrong direction remain. The improvement is reproducing the fixed decisions with fewer queries while fitting the resource budgets.

### Separate optimizations excluded from this evaluation

Merging LoRA into backbone weights and requantizing can eliminate separate A/B products, but changes numerical results and probabilities. A separate three-input experiment reduced instructions by approximately 13.21–13.52%. This proposal evaluation retained the original INT8 backbone plus separate F32 LoRA pack. A Wasm filename containing merged does not mean this evaluation used a merged pack.

Some deeper Strassen, Winograd, and Hopcroft/Kerr candidates reduced multiplication counts but increased transform or reconstruction instructions and were rejected. Theoretical multiplication reductions are not counted directly as IC performance improvements.

Pay-per-call update inference and extraction into a common runtime are separate work. The 32/50-query results do not describe update call counts or later builds. Mainnet production operation, costs, and certification of query responses are outside this report's verification scope.

## Threshold and future evaluation policy

The 0.6 threshold applies to an uncalibrated score from the A/B logit difference. It does not mean a 60% probability of correctness.

Using the same logits with a threshold of 0.5 increases reject labels among the 32 conditionally reviewed reject cases from 10 to 28. It also labels the 16 minimum-lock changes from 1 to 2 days as reject. These proposals already stop as hold, so changing them to reject does not improve pass prevention. Lowering the threshold merely to reduce hold is unnecessary for the current purpose.

Future evaluation should prioritize approve passes among proposals independently confirmed dangerous. It should also measure stops among proposals confirmed safe or acceptable, to quantify the additional review burden. This set has no independent human ground truth, so general miss rates and false-stop rates remain unmeasured.

Priorities are checking non-numerical risks in #656 and #659, and testing pass prevention on other unused proposals. Holding everything would prevent passes, so stopped counts alone cannot establish improvement. The filter should also be evaluated on its ability to pass confirmed acceptable proposals.

## Evidence

Portable copies are retained in the [evidence index](evidence/proposal-filter-20261007/README.md). [Provenance](evidence/proposal-filter-20261007/provenance.json) records original/export SHA-256 hashes, path normalization, and JSON projections. These are shared copies of retained results, not fresh inference or an independent execution attestation. Some linked historical evidence documents remain in Japanese.

- [Fresh reassessment of all 161 proposals](evidence/proposal-filter-20261007/RESULT.md)
- [Labels, routes, and reasons for all 161 proposals](evidence/proposal-filter-20261007/DECISIONS.md)
- [Independent execution verification record](evidence/proposal-filter-20261007/verification.json)
- [Original proposal evidence review and threshold comparison](evidence/proposal-filter-20261007/REVIEW.md)
- [Old/new values and conditional review for each proposal](evidence/proposal-filter-20261007/review.json)
- [Pinned model and tokenizer information](../MODEL_LOCK.json)
- [Quantized pack manifest](evidence/proposal-filter-20261007/pack-manifest.json)
- [Prefix and suffix execution plan](evidence/proposal-filter-20261007/plan.json)
- [Execution modules and reuse policy](evidence/proposal-filter-20261007/execution-policy.json)
- [Fresh inference and preparation query totals](evidence/proposal-filter-20261007/fresh-summary.json)
- [Example execution record for a 128-token input](evidence/proposal-filter-20261007/run-000.json)
- [IC optimization methods and history](COMPUTE_REDUCTION.md)
- [Frozen Wasm build, features, and kernel substitutions](evidence/proposal-filter-20261007/build.json)
- [Query reduction and intermediate state comparison for the same 18 inputs](evidence/proposal-filter-20261007/OPTIMIZATION.md)
- [Independent comparison of the same 18 inputs](evidence/proposal-filter-20261007/optimization-verification.json)
- [32-query scheduling and verification history](QUERY32_PROGRESS.md)
- [Separate LoRA merge and requantization experiment](MERGED_ADAPTER_QUERY32.md)

The earlier fragment-based experiment that returned hold for every case did not constitute proposal-level assessment and is excluded as evidence of filter performance.
