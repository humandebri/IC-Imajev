# Client-held ordinary-query protocol v1

Candid `step(blob)` carries `u32 little-endian header_length | UTF-8 JSON header | contiguous little-endian F32 values | SHA256(previous bytes)`. Header contains version1, model lock SHA256, pack SHA256, input SHA256, step counter, operation, tensor name, dims, scalars. Maximum blob2,000,000 bytes, header16,384 bytes. Legacy F32 activation limit450,000 elements; optional lossless bf16-exact limit900,000 elements subject to the same encoded byte limit. Finite values required. Reply increments step and keeps identity/op/dims/scalars. JSON F32 scalars must be compared after casting to F32; their decimal representation may change.

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

See [ACTIVATION_INT8.md](ACTIVATION_INT8.md) for the block256 codec, F32 state/gate tail, and `delta_heads_bf16`, `attention_heads_bf16`, `rope_heads` layouts. Only INT8 delta-head requests may carry up to1,200,000 floats; the encoded blob limit remains2,000,000 bytes. Other INT8 payloads remain limited to900,000 floats. The raw `int8_matmul` prototype also underlies the opt-in `lora_integer`/`linear_integer_bf16` full graph. See INTEGER_ARITHMETIC.md for arithmetic quantization and separate judgment validation.

## Exact grouped projection

`lora_grouped` carries dims `[tokens, virtual_tile_width, cols, row_start, total_rows]`, aux A/B and scale. At most3 original tiles and1,350M MACs; legacy projection remains450M. The reply concatenates token-major virtual tiles padded with zeros to256-float boundaries; the client strips padding. This preserves the old INT8 block quantization. See [EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md). Owner-only step also accepts scalar Attention reference operations for same-Wasm numerical tests.


## Larger lossless element operations

Only binary flat `add_bf16`, `swiglu_bf16`, and `attention_gate` may carry one length up to450,000 with exactly twice that many input floats. Other dimensions retain262,144. Optional `--compact-lossless` requires `bf16-exact`; it increases independent normalization, convolution-channel and element chunks without changing INT8 transport block boundaries.

## Lossless SIMD and fused add/norm

`bf16-exact` frames keep the identical bitmap, payload order, SHA-256, and canonical checks. All-BF16 frames have a SIMD pack/unpack fast path; mixed F32 frames retain generic processing. Integer projections use SIMD max/divide/RNE/clamp with unchanged block256 scales. Non-final input token chunks are aligned down to8 when the blob/token cap forces a split. New sessions record `integer_scheduler=aligned-token8-v1`; use a fresh directory for previous integer journals.

`add_norm_bf16` has dims `[tokens,width]`, exactly two input arrays (`residual`, `branch`), one norm tensor and positive finite epsilon. It returns two arrays: BF16 residual sum then BF16 RMSNorm of that rounded sum. Each array has at most450,000 floats. `--fuse-add-norm` records a new session flag and requires lossless wire. The next layer's input norm is attributed to the preceding layer's final fused query. Partial graphs return the original hidden state.

Owner-only `profile_step(blob)` returns the ordinary measurement plus named inclusive `(span,instructions,calls)` tuples. It is enabled only in `--features instruction-profile` builds, otherwise returns an error. Nested spans must not be added to parents; profiling compiler effects and counter overhead are reported separately. Default full inference compiles all span hooks away. See [BOTTLENECKS.md](BOTTLENECKS.md).

## Fused integer MLP

`mlp_gate_up_integer` uses dims `[tokens,rows,cols,row_start]`, gate weight as tensor, the corresponding same-layer up weight as the sole aux tensor, and one adapter scale. The manifest binds each base INT8 tensor and its separate F32 LoRA A/B. Both projections preserve their original BF16 boundaries before the existing BF16 SwiGLU. Block256 input quantization is shared; LoRA uses the original F32 input.

The combined base+LoRA work limit is4,000,000,000 MACs, rows are multiples of8, cols multiples of256, tokens at most512, each base weight tile at most30M bytes, output at most900k floats, and encoded blobs at most2,000,000 bytes. Metadata and bounds are checked before loading weights. The client keeps all intermediate state and returns only the SwiGLU product. All inference remains ordinary query.

`--fuse-mlp` requires integer arithmetic and `bf16-exact` transport. `mlp_scheduler=work-budget-v2` sessions use the requested row cap within combined work/weight/blob limits; previous journals require a fresh directory. An instruction-limit failure halves the width on an8-row boundary without advancing progress. See [MLP_FUSION.md](MLP_FUSION.md) for measured bounds and precision comparisons. Expanded integer-dot trees preserve exact I32 sums, block order and F32 scaling.

## Client-held shared prefix

No new canister method or persistent query state is introduced. The separate `run_prefix_canister.py` prepares a real45-token prefix via ordinary query, then optionally uses `PrefixTextGraph` to process matching suffixes. Prefix metadata binds model, pack, Wasm and graph source hashes, exact token IDs, and all64 file hashes. Conv/DeltaNet states remain exact F32; KV/hidden retain their existing values. Absolute RoPE positions start at the prefix length. The original causal attention kernel receives combined KV and zero prefix queries, and only suffix outputs are used. Readout/calibration remain unchanged.

An initial cache requires additional ordinary queries and client storage. It is never silently enabled in the standalone runner. Cache/session mismatch fails before graph queries. Tests cover model/pack/Wasm/source/file/position mismatch. See [DIRECTIONS.md](DIRECTIONS.md) for first-question costs and measured inputs.

Base integer projection weights now stay INT8 in memory; each shared8-byte load is sign-extended in SIMD registers, with validated ranges. Output tiles are8 rows; token tiles and block256 arithmetic order are preserved. This changes no wire format or additional quantization.
