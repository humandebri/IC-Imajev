# Client-held ordinary-query protocol v1/v2

Candid `step(blob)` carries `u32 little-endian header_length | UTF-8 JSON header | contiguous little-endian F32 values | SHA256(previous bytes)`. Header contains version1, model lock SHA256, pack SHA256, input SHA256, step counter, operation, tensor name, dims, scalars. Maximum blob2,000,000 bytes, header16,384 bytes. Legacy F32 activation limit450,000 elements; optional lossless bf16-exact / bf16-block256-exact-v1 limit900,000 elements subject to the same encoded byte limit. Finite values required. Reply increments step and keeps identity/op/dims/scalars. JSON F32 scalars must be compared after casting to F32; their decimal representation may change.

Version2 optionally replaces only the final SHA256 digest with the 32-byte unkeyed BLAKE3 digest. It requires the `experimental-blake3` runtime feature and `--frame-checksum blake3` on the client; version1 remains supported/default. Both algorithms cover the same complete header and payload. The bounded header is parsed to select the algorithm before checksum verification; payload decoding and execution happen only after verification. Unknown versions are rejected. Model/pack/input/file/module/source hashes remain SHA256. See [FRAME_CHECKSUM.md](FRAME_CHECKSUM.md).

The canister checks pack/model identity, immutable prepared tensor table, bounds, finite values and checksums. Input hash and step are client continuity labels, not proof of honest computation. Every method checks the install-time owner. The client may alter its own intermediate states; query results are not certified.

For matmul dims=`[tokens, output_rows, input_cols]`, all tensor rows must match. dims=`[tokens, tile_rows, input_cols, row_start]` reads a contiguous output-row tile; INT8 row scales are fetched separately. Weights F32, BF16, symmetric per-row INT8 are supported. Matmul work above120,000,000 multiply-adds is rejected before computation. This operation cap alone is not a proof that every shape stays below4 billion instructions.

Other operations: rms `[rows,width]` + epsilon; conv `[tokens,channels,kernel]`; causal attention `[tokens,head_dim]` with q/k/v concatenated; delta `[tokens,key_dim,value_dim]` with q/k/v/decay/beta/state concatenated. Delta state layout `[value_dim,key_dim]`, F32. Reply contains output followed by continued state. Legacy conv starts from zeros; conv_state carries the previous kernel−1 input tokens. TextGraph connects all 32 layers and JournalTransport validates client-held saved requests/replies before resuming; see FULL_INFERENCE.md.

prepare(text manifest) allocates stable pages and freezes tensor identity at seal. upload(offset, blob, chunk_sha256) is sequential, accepts byte-identical repeats, and updates a running pack digest. seal checks completeness and the whole pack SHA256. A sealed pack cannot be replaced. Upgrade stores metadata after weight pages and restores it with the owner. An incomplete upload restarted across upgrade resets upload progress to0 because SHA256 internal state is not serialized.

`decision(blob, options)` accepts only this model's dedicated readout request, computes all256 rows, selects `options.len()+1` rows (unknown last), uses the fixed calibration temperature and returns `opt text`, probabilities, unknown_probability, abstained, raw_logits. It supports2–7 unique non-reserved strings. This covers Choice; ordinal/Score/rubric binding is not implemented.

The local bridge only accepts loopback URLs. It reads a local-only secp256k1 PEM identity from ignored artifacts/. Transport records Candid request and reply bytes; it does not measure outer HTTP/CBOR envelopes. Binary requests persist for bounded retries. Transport.project halves row tiles on instruction/work limit failures without advancing client state. Other operation failures stop without switching inference to update.

## 効率化拡張

`Request.aux`は省略可能な最大2 tensor名。`lora_project`では`[LoRA A, LoRA B]`、dims=`[tokens,output_rows,input_cols,row_start?]`、scalars=`[scale]`を指定する。model/pack identityは従来通り必須。出力はtoken-major BF16値のF32表現。Aの全rankを読み、baseとBは同じrow rangeを読む。rank上限256、総積和数450M。matmul単体の120M上限は維持する。

`matmul`はSIMD、`matmul_reference`は旧scalar比較用。両方とも行tileに対応。`decision_fast`は旧`decision`と同じ要求形式・型付き返答だが、候補数＋unknown行のみ計算する。計算queryはstateを永続化しない。

`Transport.project`は独立token線形射影だけを分割する。成功返信までprogressを進めず、命令/work上限エラー時に行幅を半減する。可逆BF16通信codecとINT8通信codecを選択できる。

Full-pack upload adds owner-only upload_chunk(offset,bytes,sha), pack_status(), hash_pack(max_bytes). Chunks are aligned at1,800,000 bytes; at most8,000,000 contiguous bytes are hashed per update. The client uploads up to16 chunks concurrently, skips matching uploaded offsets, and checks whole-pack hash before inference. Sealed metadata survives upgrade; incomplete upload progress resets.

Full graph also uses embed, linear_bf16, rms_bf16/rms_scaled, conv_state, delta_gates/delta_bf16, gated_norm, rope, attention_bf16, attention_gate, swiglu_bf16, add_bf16. BF16 boundaries are encoded as exact F32 values; F32 recurrent states remain F32. Activation INT8 transport is implemented separately from weight INT8; see ACTIVATION_INT8.md.

## 可逆BF16通信codec

Headerの省略可能な`encoding: "bf16-exact"`を指定する。payloadは`u32 little-endian count | ceil(count/8) bytes bitmap | values`。bitmapのbit i（LSB first）が0ならBF16の上位16 bitをu16 little-endianで保存し、1なら元のF32 u32を保存する。元F32の下位16 bitが0である値だけ2 byteへ短縮し、量子化・丸めは行わない。F32 recurrent stateとdecayの例外値は4 byteで保存する。SHA256はencoded header/payload全体を対象にする。

decoderはcount、payload完全長、bitmap未使用bit、finite値、非正規なF32例外（下位16 bitが0）を検証する。旧encoding省略のF32形式も受け付ける。replyは同じcodecで返し、identity比較にencodingを含む。2 MBを超えるmixed payloadは、900,000要素以下でも拒否する。

全層runnerは450M積和・最大1,536出力行・最大132 tokenで分割する。入力/出力900,000要素を上限とし、線形射影のtoken幅は可能なら4の倍数へ揃える。各SIMD laneの積和順を保ち、余りtokenは独立に処理する。重みINT8や中間状態INT8とは別の可逆通信形式である。

## INT8 transport and head fusion

### Lossless block256 transport

`encoding: "bf16-block256-exact-v1"` is a separate lossless wire format. Its payload is `u32 little-endian count | ceil(ceil(count/256)/8) bytes bitmap | block data`. Each LSB-first bitmap bit describes up to 256 consecutive values. Bit 0 stores every value as its exact BF16 upper 16 bits; bit 1 stores every value as the original F32 bits. A full F32 block must contain at least one value whose low 16 bits are nonzero. Partial final blocks contain only their actual elements. Unused bitmap bits must be zero; trailing or truncated payloads are rejected.

The count remains bounded to 900,000 floats and the complete encoded frame, including header and checksum, to 2,000,000 bytes. Finite-value, checksum, identity and continuity checks are unchanged. This format does not quantize or round activations or recurrent states. A mixed block can be larger than the old per-element format when exceptions are sparse. Encoders reject oversized frames rather than silently changing precision.

All existing operations requiring lossless wire accept either `bf16-exact` or this block format. Replies retain the request encoding. Use `--wire-codec bf16-block256-exact-v1` with a fresh client session and a freshly prepared prefix bound to that session/module. The legacy format and the separate lossy INT8 transport remain supported; this change does not increase operator work or query instruction limits.

See [ACTIVATION_INT8.md](ACTIVATION_INT8.md) for the block256 codec, F32 state/gate tail, and `delta_heads_bf16`, `attention_heads_bf16`, `rope_heads` layouts. Only INT8 delta-head requests may carry up to1,200,000 floats; the encoded blob limit remains2,000,000 bytes. Other INT8 payloads remain limited to900,000 floats. The raw `int8_matmul` prototype also underlies the opt-in `lora_integer`/`linear_integer_bf16` full graph. See INTEGER_ARITHMETIC.md for arithmetic quantization and separate judgment validation.

## Exact grouped projection

`lora_grouped` carries dims `[tokens, virtual_tile_width, cols, row_start, total_rows]`, aux A/B and scale. At most3 original tiles and1,350M MACs; legacy projection remains450M. The reply concatenates token-major virtual tiles padded with zeros to256-float boundaries; the client strips padding. This preserves the old INT8 block quantization. See [EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md). Owner-only step also accepts scalar Attention reference operations for same-Wasm numerical tests.


## Larger lossless element operations

Only binary flat `add_bf16`, `swiglu_bf16`, and `attention_gate` may carry one length up to450,000 with exactly twice that many input floats. Other dimensions retain262,144. Optional `--compact-lossless` requires either lossless encoding; it increases independent normalization, convolution-channel and element chunks without changing INT8 transport block boundaries.

## Lossless SIMD and fused add/norm

`bf16-exact` frames keep the identical bitmap, payload order, SHA-256, and canonical checks. All-BF16 frames have a SIMD pack/unpack fast path; mixed F32 frames retain generic processing. Integer projections use SIMD max/divide/RNE/clamp with unchanged block256 scales. Non-final input token chunks are aligned down to8 when the blob/token cap forces a split. New sessions record `integer_scheduler=aligned-token8-v1`; use a fresh directory for previous integer journals.

`add_norm_bf16` has dims `[tokens,width]`, exactly two input arrays (`residual`, `branch`), one norm tensor and positive finite epsilon. It returns two arrays: BF16 residual sum then BF16 RMSNorm of that rounded sum. Each array has at most450,000 floats. `--fuse-add-norm` records a new session flag and requires lossless wire. The next layer's input norm is attributed to the preceding layer's final fused query. Partial graphs return the original hidden state.

Owner-only `profile_step(blob)` returns the ordinary measurement plus named inclusive `(span,instructions,calls)` tuples. It is enabled only in `--features instruction-profile` builds, otherwise returns an error. Nested spans must not be added to parents; profiling compiler effects and counter overhead are reported separately. Default full inference compiles all span hooks away. See [BOTTLENECKS.md](BOTTLENECKS.md).

## Fused integer MLP

`mlp_gate_up_integer` uses dims `[tokens,rows,cols,row_start]`, gate weight as tensor, the corresponding same-layer up weight as the sole aux tensor, and one adapter scale. The manifest binds each base INT8 tensor and its separate F32 LoRA A/B. Both projections preserve their original BF16 boundaries before the existing BF16 SwiGLU. Block256 input quantization is shared; LoRA uses the original F32 input.

The combined base+LoRA work limit is4,500,000,000 MACs, rows are multiples of8, cols multiples of256, tokens at most512, each base weight tile at most30M bytes, output at most900k floats, and encoded blobs at most2,000,000 bytes. Metadata and bounds are checked before loading projection weights. The client keeps all intermediate state and returns only the SwiGLU product. All inference remains ordinary query.

`--fuse-mlp` requires integer arithmetic and `bf16-exact` transport. `mlp_scheduler=work-budget-v2` sessions use the requested row cap within combined work/weight/blob limits; previous journals require a fresh directory. An instruction-limit failure halves the width on an8-row boundary without advancing progress. See [MLP_FUSION.md](MLP_FUSION.md) for measured bounds and precision comparisons. Expanded integer-dot trees preserve exact I32 sums, block order and F32 scaling.

## Client-held shared prefix

No new canister method or persistent query state is introduced. The separate `run_prefix_canister.py` prepares a real45-token prefix via ordinary query, then optionally uses `PrefixTextGraph` to process matching suffixes. Prefix metadata binds model, pack, Wasm and graph source hashes, exact token IDs, and all64 file hashes. Conv/DeltaNet states remain exact F32; KV/hidden retain their existing values. Absolute RoPE positions start at the prefix length. The original causal attention kernel receives combined KV and zero prefix queries, and only suffix outputs are used. Readout/calibration remain unchanged.

An initial cache requires additional ordinary queries and client storage. It is never silently enabled in the standalone runner. Cache/session mismatch fails before graph queries. Tests cover model/pack/Wasm/source/file/position mismatch. See [DIRECTIONS.md](DIRECTIONS.md) for first-question costs and measured inputs.

Base integer projection weights now stay INT8 in memory; each shared8-byte load is sign-extended in SIMD registers, with validated ranges. Output tiles are8 rows; token tiles and block256 arithmetic order are preserved. This changes no wire format or additional quantization.

## Compact head transport

`--compact-heads` enables two lossless-only operations through the existing ordinary `step(blob)` query. Session metadata binds `compact_heads=gqa-shared-terminal-delta-v1` and `terminal_states_retained`. A fresh journal is required. There is no Candid method change.

`gqa_heads_bf16` uses `[tokens,width,query_heads]` and a head-major payload of all Q, then unique K, then unique V. One KV head serves four query heads. Supported query head counts are1,2,4,8,12,16, with `ceil(query_heads/4)` KV heads. Chunk starts must align to the corresponding global KV group; the client routes the matching KV slice. Tokens≤512, width≤256, and `query_heads*tokens*(tokens+1)/2*width≤75,000,000` bound work in addition to the blob/float limits. Each head uses the unchanged causal arithmetic. The client chooses the largest fitting chunk; at132/147/512 tokens the tested chunks are16 /12+4 /eight groups of2.

`delta_terminal_bf16` has the same dims and input as `delta_heads_bf16`, including exact initial state and gates, but returns only head-major token outputs. It is used only for the last token chunk of a terminal inference. Earlier chunks use the state-returning operation. Prefix preparation always retains all states. Terminal runs save conv states and KV for verification but omit the24 Delta state arrays, so their output directory is not a continuation cache. Client retry journals still retain the inputs needed to resume unfinished work.

Both operations reject lossy wire encodings. Arithmetic, weight precision, BF16 boundaries, LoRA, readout and calibration are unchanged. See [COMPACT_HEADS.md](COMPACT_HEADS.md) for full-run results and scope.

### Delta stage fusion (lossless BF16 only)

- `delta_stage_bf16`: dims `[tokens, heads, first_head, keep_state]`; tokens 1..512, even heads 2..16, even first_head, first_head+heads<=32, keep_state 0/1. Tensor is the layer's `.linear_attn.conv1d.weight`; aux is its `.norm.weight`. Input consists of token-major `(tokens+3)×(heads×256)` raw convolution window (Q half-heads, K half-heads, V heads), token-major z `[tokens,heads,128]`, decay and beta `[tokens,heads]` each, and head-major F32 state `[heads,128,128]`. Reply is gated `[tokens,heads,128]` plus F32 final state iff keep_state=1. All constituent primitives preserve existing BF16 rounding and accumulation order. Client retains the final three raw convolution rows.
- `delta_gates_integer`: dims `[tokens]`, 1..256; tensor `.linear_attn.in_proj_a.weight`, no aux/scalars. Input `[tokens,2560]`; derives matching in_proj_b, A_log, dt_bias. Executes original two INT8 projections with BF16 output boundaries followed by delta_gates. Reply is all decay then all beta. A/B projection LoRA is rejected by this fixed-model operation.
- Combined MLP work guard is 4.5G; opt-in client scheduling uses it only for <=89 tokens and retains 4.0G above that. This guard is an operation estimate, not a change to IC's query instruction limit. Adaptive fallback remains available and failed calls are written to `failures.jsonl`.

### Terminal readout and suffix Attention

- `gqa_suffix_bf16`: dims `[query_tokens,width,heads,prefix]`. Query rows occupy absolute positions `prefix..prefix+query_tokens`; all `prefix+query_tokens` K/V rows are sent. Q is head-major, then unique K and V groups as in GQA. Bounds: positive query_tokens, total<=512, width<=256, supported heads 1/2/4/8/12/16, suffix causal work<=75M, lossless BF16 only. Output contains query_tokens rows per head; output indexing is relative even though the causal boundary is absolute.
- `terminal_mlp_integer`: dims `[1,2560]`, tensor fixed to `model.language_model.layers.31.post_attention_layernorm.weight`, no aux/scalars. Input is residual hidden followed by attention output, each2560 floats. Runs original residual/norm, INT8 gate/up+F32 LoRA+SwiGLU, down projection, final residual/norm. Reply contains unnormalized last-layer hidden then normalized readout input, each2560 floats. Decision/calibration remains in the separate decision query.
- `delta_input_integer`: dims `[tokens]`, 1..96; tensor `.linear_attn.in_proj_qkv.weight`, no aux/scalars. Input `[tokens,2560]`, output all QKV `[tokens,8192]`, decay `[tokens,32]`, beta `[tokens,32]`. Keeps original QKV F32 LoRA and BF16 boundaries, derives A/B gate tensors. Client enables only if the whole QKV fits the configured token, row and work caps.

### Residual/norm fusion into MLP

- `mlp_add_norm_integer`: dims `[tokens,rows,cols,row_start]`, same gate tensor and original MLP bounds. Aux contains the corresponding up tensor followed by the same layer's `post_attention_layernorm.weight`; scalars are adapter scale and positive finite epsilon. Input is all residual hidden followed by attention output, each `[tokens,cols]`. Computes BF16 residual addition and BF16 RMSNorm, then the unchanged integer gate/up and F32 LoRA/SwiGLU. Reply contains only the SwiGLU tile. Normalization is repeated when the projection must use multiple row tiles.
- `add_norm_chain_bf16`: dims `[tokens,width]`, one norm tensor, no aux, positive finite epsilon. Input is three token-major arrays: residual before attention, attention output, then MLP output. Computes `BF16(BF16(residual+attention)+MLP)` and returns that hidden followed by its BF16 RMSNorm. Both rounding boundaries are preserved. Tokens are bounded to 512, total input floats to 900,000, and encoded blob bytes to 2,000,000.
- Both require either lossless encoding, remain owner-only ordinary `step(blob)` queries, and introduce no persistent inference state or Candid change. `--fuse-mlp-norm` requires `--fuse-mlp --fuse-add-norm` and binds a fresh session flag. The client uses the previous path when three residual arrays do not fit, or when a partial graph's last layer must return the original hidden without next-layer norm. Terminal layer 31 retains its existing single-token fused operation.

### Shared RoPE and fused Q/K normalization

`norm_rope_heads_bf16` has dims `[tokens,width,rotary_width,absolute_offset,heads]`, one norm tensor, no aux, and scalars `[epsilon,theta]`. It requires lossless wire, positive finite epsilon/theta, tokens 1..512, width 1..256, even positive rotary_width<=width, offset<=262144, heads 1..16, and exactly `tokens*width*heads` input floats, at most900,000. Input/reply are head-major `[heads,tokens,width]`. Norm weight contains exactly width values. It applies the original BF16 RMSNorm per row, then the original partial RoPE and BF16 rounding; nonfinite output is rejected. The complete encoded frame remains bounded to2,000,000 bytes.

`--fuse-norm-rope` records `rms-rope-shared-angles-v1` in a fresh client session. It removes the separate Q/K normalization queries; no question-dependent math runs on the client. Raw `rope`/`rope_heads` retain their layouts and share the exact frequency/position calculation internally. Fixed A_log exponentials are similarly shared across tokens within each delta-gate query. There is no new persistent inference state, Candid method, precision setting, or query instruction limit.


### Immutable weight preparation and borrowed reads

The additive owner-only methods are `warm_weights(name : text)` (update returning `Result<WeightCacheInfo, text>`), `clear_weight_cache()` (update), and `weight_cache_status()` (query). `WeightCacheInfo` contains `bytes : nat64`, `names : vec text`, and `preparation_instructions : nat64`. Sealed fixed INT8 tensors with validated row scales and original F32 tensors with finite values may be prepared; the cache contains no question data or inference state. Preparation is idempotent for an already cached tensor. Upgrade resets this optional heap cache while retaining the stable model pack.

The reader returns an immutable borrowed byte slice for a cached bounded range and allocates/stable-reads only on a miss. The runtime views base bytes as i8 without copying; values, scales, integer dots, F32 order and BF16 boundaries are unchanged. `stable_read_bytes` now counts physical stable reads performed by the canister handler, while the generic runtime still returns logical tensor bytes. These bytes are separate from Candid communication. `decision_fast` still does not expose this read counter.

The default cache budget is256 MiB; `experimental-full-weight-cache` raises it to4,080,000,000 bytes with a128 MiB per-tensor cap. Preparing the full dense model requires the selected local canister's4 GiB heap limit. Query limits, request frame limits, owner checks and client-held intermediate state remain unchanged. See [BORROWED_WEIGHTS.md](BORROWED_WEIGHTS.md) for preparation cost and full-graph measurements.


F32 preparation decodes little-endian values and checks finiteness once, storing an immutable `PreparedF32` backed by `Rc<[f32]>`. Prepared row tiles share the same allocation and the runtime borrows this validated typed view. The existing byte-reader public APIs still work through a byte-only wrapper. Finite input/output checks, shape/range/work checks and all numerical precision settings remain intact. The concrete byte-view implementation supports the selected little-endian Wasm/native targets. `prepare_weight_cache.py --include-f32` prepares721 tensors totaling4,065,416,192 bytes; no question state is stored by these updates. See [PREPARED_F32.md](PREPARED_F32.md) for the tested memory margin and ordinary-query measurements.


### Experimental prepared Strassen coefficients

Only `experimental-strassen-prepared` permits `warm_weights(source+".strassen_i8")` for the diagnostic derived dtype `strassen-i8-v1`. The packed reduction feature includes preparation support at both canister and runtime levels. Preparation validates all derived I16 coefficients against the original INT8 source and stores positive finite row scales, bound to the sealed model/pack/source. Cache accounting includes derived coefficients; preparation is idempotent and upgrades clear it. A prepared `linear_strassen_bf16` query retains input/metadata/range/work checks and physical read accounting, with no question state retained. The ordinary full-model feature does not enable this experimental operation. The tested partial candidate is slower than ordinary INT8 and is not adopted. See [STRASSEN_PREPARED.md](STRASSEN_PREPARED.md) for measured scope and feature-dependency failure.


### Experimental client-held projection reuse

`experimental-projection-reuse` enables `lora_integer_capture` and `lora_integer_reuse` with the same four dims and metadata as `lora_integer`, lossless encoding only. Capture appends token-major quantized integer values `[n,cols]`, original F32 block scales `[n,cols/256]`, then original F32 LoRA-A results `[n,rank]` after its BF16 output tile `[n,rows]`. Reuse accepts just that state and returns the next output tile; rank comes from the sealed A/B tensors. The client holds state, the canister validates the untrusted state every request, and no question-dependent state is persisted. Combined capture output remains at most900,000 values and the unchanged2,000,000-byte frame cap applies. The next row boundary may be reduced to fit the returned state. This experiment has not been integrated into the production scheduler: it is slower for132 tokens and increases communication. See [PROJECTION_REUSE.md](PROJECTION_REUSE.md).


### Exact INT8 projection-state codec

The opt-in `projection-block256-exact-v1` codec is feature-gated by `experimental-projection-reuse`. Only capture/reuse projection operations accept it, with dims `[tokens,rows,cols,row_start,rank]` and rank checked against sealed A/B tensors. Tag0 wraps the existing lossless block256 payload for capture input/reuse output. Tag1 stores capture output as BF16 tile, then exact signed INT8 values, then raw F32 scales/A products; tag2 stores just the INT8/raw F32 state. The existing full-frame identity/progress header and SHA256 remain. Bounds, canonical direction, lengths, finite positive scales, reserved integer code, and ambiguous shapes are checked. Integer negative zero is rejected rather than silently normalized; F32 A signed zero/subnormal bits remain exact.

`--reuse-projection-inputs` activates the production client path only when capturing state does not increase the scheduled query count. It records `client-held-int8-f32-a-v1` in a fresh session. Journal replay reconstructs state from the saved first response; instruction failure does not commit a failed tile/state. The temporary codec switch is restored on success or error. Full-graph verification passes all five reference conditions; measured cold instruction reduction is0.09652% with0.2961% more Candid traffic and unchanged485 queries. It does not reduce the prepared-prefix305-query path. See [PROJECTION_CODEC.md](PROJECTION_CODEC.md).

### Direct reconstruction of projection reuse input

With `experimental-projection-reuse`, `decode_query` validates the existing tag2 frame and reconstructs private `PreparedProjection` directly from INT8 bytes into immutable I16 lanes, bypassing the intermediate F32 integer array. Positive finite scales, finite A products, reserved integer codes, shape/length limits and checksums are still checked on every client-held input. The prepared value is bound to all request identity and arithmetic fields; altered metadata is rejected before weight reads. Wire format, Candid API, quantization and arithmetic order are unchanged. See [DIRECT_PROJECTION.md](DIRECT_PROJECTION.md) for the measured scope and full-model comparisons.

### MLP projection capture/reuse

`mlp_gate_up_capture` and `mlp_gate_up_reuse` extend the exact projection codec with two F32 A products. Dimensions remain `[tokens,rows,cols,row_start,rank]`; the state is INT8 input, block256 F32 scales, then gate A and up A arrays. Both sealed A/B shapes and F32 precision must match. The first capture output is BF16 SwiGLU plus this exact state; reuse returns the next BF16 output tile. Work, frame and value bounds remain checked, with no question state retained in the canister. The client enables it only without add/norm fusion and when capture does not increase the row query count. Session policy is `client-held-int8-f32-a-and-mlp-v2`; failed attempts do not commit progress/state, and codec restoration uses try/finally. See [MLP_REUSE.md](MLP_REUSE.md) for scope and measurements.

Optional `experimental-attention-fusion` adds `attention_kv_integer` `[tokens,position_offset]` and `attention_q_gqa_integer` `[query_tokens,total_tokens,total_tokens-query_tokens,first_head,heads]`. Both preserve the original BF16/integer/F32-LoRA arithmetic. KV shares validated activation quantization; Q evaluates norm/RoPE, full-history causal GQA and gate internally. Fixed hidden2560, rank64, Q16/KV4 geometry; aux/scalars empty. Heads and first_head align to GQA groups of4, all head/history bounds are checked, and the same frame2MB/900K limits apply. See [ATTENTION_FUSION.md](ATTENTION_FUSION.md).

Optional `experimental-delta-projected` adds capture/reuse `[tokens,heads,first_head,keep]` with tensor `<linear_attn>.in_proj_qkv.weight`, no aux/scalars. Capture input is original normalized hidden, head-group conv history and F32 recurrence. Reuse input replaces hidden with client-held exact quantized integer values, raw F32 scales/QKV-A/Z-A/gates. Replies carry gated output, raw conv history, optional recurrence and capture-only prepared state. All tokens/heads remain included; frame2MB/900K bounds and finite/scale/integer checks remain. Only bounded internal scratch may exceed the wire value count. See [DELTA_PROJECTED.md](DELTA_PROJECTED.md).

## 入力バッファの一回書き込みと固定RoPE準備

量子化・整数状態復元・返信追加は未初期化spare capacityへ全active要素を書いた後にlengthを確定し、token paddingだけゼロ初期化する。整数性・範囲・有限性検査と既存wireを維持する。`experimental-prepared-rope`はownerの`warm_weights`で固定theta=10,000,000・rotary64・512位置のsin/cosを一度準備する。queryは表を参照し、範囲外/異なる設定では従来計算へ戻る。`weight_cache_status.rope_bytes`は後方互換の`opt nat64`で、表の準備後131072、feature無効はNone。表はweight cacheの`bytes`と別計上し、ownerのclear/upgradeで消える。準備と全層検証は`--require-prepared-rope`で確認する。client-held推論状態・通常query・精度設定は維持する。[実測と再現手順](ACTIVATION_BUFFERS.md)。

`experimental-prepared-output-pairs`は固定INT8重みをK4/2出力の配置へowner準備updateで置換する。元weight値・scale・payload容量は維持し、元配置の全コピーを追加しない。`weight_cache_status.paired_weight_bytes`は後方互換の`opt nat64`で、変換済みINT8 weight byte数（scaleを除く）を返す。これは`bytes`に含まれる内数で、別の追加cache量ではない。feature無効はNone、clear後は0、upgrade後は再準備が必要。通常queryのwire形式は従来のまま。入力変換の再利用はquery内に限定し、質問状態をcanisterへ永続化しない。準備・全層検証は`--require-output-pairs`で対応を確認する。[実装・検証](OUTPUT_PAIRS.md)。

## INT8出力tile16

canister feature `experimental-column16`は16で割り切れる投影行数を16行tileで処理し、それ以外は既存8行tileへ戻す。token tileは最大32。INT8重みのblock256展開を明示し、量子化・整数dot・F32 scale/加算順序を維持する。Candid/wire、clientの分割方針、上限、追加精度設定は変えない。[全層比較](COLUMN16.md)。

## Delta後半と出力投影の統合

feature `experimental-delta-finish`とclient `--fuse-delta-finish`で`delta_project_finish`を有効にする。dims `[tokens,16,16,keep]`、tokens1..90、既存後半reuse入力に前半16-head gated出力をtoken-majorで追加。返信は2560幅の投影値、後半conv履歴、keep時のみ後半F32状態。元のINT8 out_proj・F32 LoRA・BF16境界を維持する。clientはcodec別入力上限を検査して統合を選び、132 tokenや容量不足は従来のreuse＋投影へ戻す。2MB/900K値/5B命令/4GiB heapを維持。[全層測定と境界](DELTA_FINISH.md)。
