# 計算量削減の手段まとめ

2026-10-03時点の実装・実測記録を、削減した処理ごとに整理した。個別の文書には比較条件、実装、再現手順、source/Wasm hashがある。本書の数値は既存記録からの転記で、今回新たにベンチマークを実行したものではない。

ここでいう「計算量削減」は主に通常queryのhandler命令数の削減を指す。行列積の計算量の次数を変える改善だけでなく、重複計算、ロード、コピー、検証、通信処理を減らす改善も含む。query数・通信量の削減と、固定処理を準備時へ移す改善は別に記録する。

## 1. SIMDとタイル配置で同じロードを共有する

| 手段 | 減らした処理 | 実測・詳細 |
| --- | --- | --- |
| F32射影の複数token SIMD | 同じ重みをtokenごとにロードする処理。各tokenを独立laneで処理し、列順の積和とBF16境界を維持 | 初期の部分QKVではscalar＋4演算からSIMD＋融合で命令85.2%減。[初期記録](OPTIMIZATIONS.md)、[全層への拡張](EXACT_OPTIMIZATION.md) |
| INT8 baseの整数dot | INT8重みをF32へ展開して行う積和を、block256のI32 dot＋F32 scaleへ変更 | 導入時の全132 tokenで1.762兆→1.022兆命令、41.97%減。**activation A8を導入した演算方式変更**であり、旧F32方式とのbit一致改善とは区別する。[INTEGER_ARITHMETIC.md](INTEGER_ARITHMETIC.md) |
| 256列dot・水平和・結果取り出しのSIMD化と展開 | accumulator更新、scalar lane取り出し、一時配列の初期化・store/load、ループ添字計算 | dot単独の段階で8468億→7769億命令。以後、定数添字展開とpacked reductionを追加。[MLP_FUSION.md](MLP_FUSION.md)、[KERNEL_UNROLL.md](KERNEL_UNROLL.md)、[PACKED_REDUCTION.md](PACKED_REDUCTION.md) |
| 重みの直接INT8ロード | 大きなI16中間配列への展開・確保を省き、必要な8 byteだけロードして符号拡張 | 出力tile8との組合せで全132 tokenの命令3.82%減。[DIRECTIONS.md](DIRECTIONS.md) |
| 16/32出力・複数tokenで入力と重み展開を共有 | 同じ入力ロード・重み展開を出力行またはtoken群ごとに繰り返す処理 | column16、token48、column32を順に検証。column32では87-token MLPの1 query化も実現。[COLUMN16.md](COLUMN16.md)、[COLUMN16_TOKEN48.md](COLUMN16_TOKEN48.md)、[COLUMN32.md](COLUMN32.md) |
| token分割の整列と均等化 | paddingによる余分な積和、少数tokenのための重み再展開 | 8境界への整列、87 tokenの48＋32＋8から44＋44への変更を実施。後続のoutput-pairsは実token数で処理し、44＋43等でpadding積和を省く。[BOTTLENECKS.md](BOTTLENECKS.md)、[BALANCED_TOKEN_TILES.md](BALANCED_TOKEN_TILES.md)、[OUTPUT_PAIRS.md](OUTPUT_PAIRS.md) |
| F32 LoRAの端数tokenもSIMD化 | 端数tokenで繰り返す重みロード。回帰する形状は従来kernelを選択 | 主suffix87で全モデル命令4.5150%減。[MATRIX_TAIL.md](MATRIX_TAIL.md) |
| Delta再帰の係数共有 | value方向で同じK/Q係数のload・splatとkeyループを繰り返す処理。decay後の中間state storeも省略 | 主suffix87で係数共有の段階は全モデル命令3.99%減。F32状態・key順積和は維持。[DELTA_SIMD.md](DELTA_SIMD.md)、[PACKED_REDUCTION.md](PACKED_REDUCTION.md)、[KERNEL_UNROLL.md](KERNEL_UNROLL.md) |

## 2. 同じ入力・状態を再計算しない

| 手段 | 減らした処理 | 実測・詳細 |
| --- | --- | --- |
| 入力量子化・LoRA A積の共有 | 同じ入力を使うQKV/gate、Attention K/V、MLP gate/up、出力行分割間のblock256量子化と元F32 A積の再計算 | 同じquery内で共有し、queryをまたぐ場合はINT8入力・scale・F32 A結果をclient-held状態で渡す。再量子化はしない。[DELTA_REUSE.md](DELTA_REUSE.md)、[PROJECTION_CODEC.md](PROJECTION_CODEC.md)、[MLP_REUSE.md](MLP_REUSE.md)、[MLP_PIPELINE.md](MLP_PIPELINE.md) |
| Q/K正規化、RoPE、固定A_logの共有 | 同じQ/Kを使うhead間のgather・RMSNorm、headごとの同一sin/cos、tokenごとの同一exp | Deltaのvalue head pairでQ/Kを共有。RoPEの周波数・positionごとのsin/cos、A_logのexpをquery内で一度だけ計算。[DELTA_REUSE.md](DELTA_REUSE.md)、[ROPE_REUSE.md](ROPE_REUSE.md) |
| 共通45-token prefixの再利用 | 毎質問の共通prefixに対するembedding・射影・conv・Delta再帰・MLP | canisterの通常queryで一度作ったhidden/conv/Delta/KV/絶対positionをclientが保持し、以後はsuffixを計算。導入時は命令30.73%減、ただしprefix準備費用は別。[DIRECTIONS.md](DIRECTIONS.md) |
| 検証済み入力を変更不能な型で保持 | 同じquery内で量子化済み配列の値域・scale・有限性を再走査する処理 | 各queryの外部入力検証は維持し、その後の重複検証・再変換を除去。主問題の命令0.560%減。[QUANTIZED_INPUT_REUSE.md](QUANTIZED_INPUT_REUSE.md)、[DIRECT_PROJECTION.md](DIRECT_PROJECTION.md) |

追加の**LoRA入力共有は全モデル検証済み・主canister未反映**。gate/upの異なるLoRA A行列へ、一度だけ並べ替えたF32入力を渡す。A積kernel単体で2.63〜4.38%減、主問題の全モデルでは0.0261%減。full MLPの上限を87→89 tokenへ拡張した変更も同じrelease Wasmで全5条件を検証し、89-token入力は98→67 queryになった。ソースへ反映済みだが、既定上限87と主canisterのmoduleは維持している。[LORA_MLP89.md](LORA_MLP89.md)。

## 3. 固定処理をモデル準備時の一度だけにする

| 手段 | 毎queryから除いた処理 | 準備・メモリとの交換条件 |
| --- | --- | --- |
| INT8/F32重みのcacheと借用 | stable read、byte復元、配列確保・コピー、固定F32重みの有限性検証 | INT8借用で主命令1.923%減、F32準備でさらに1.038%減。固定重みcache約4.065 GBを使用。[BORROWED_WEIGHTS.md](BORROWED_WEIGHTS.md)、[PREPARED_F32.md](PREPARED_F32.md) |
| 固定RoPE表 | 周波数・sin/cosの生成 | owner準備updateで一度生成。量子化・復元・返信bufferの全域ゼロ初期化も除去し、paddingだけ初期化。[ACTIVATION_BUFFERS.md](ACTIVATION_BUFFERS.md) |
| BF16活性化関数の完全lookup | 要素ごとのsigmoid/SiLU/softplus等のexp・除算・log | 有限BF16入力65,280通りの元の出力bitをWasm自身で生成。非BF16等は元の演算へfallback。表1 MiB、主命令2.6956%減。[PREPARED_ACTIVATION.md](PREPARED_ACTIVATION.md) |
| INT8重みのK4/2出力配置＋query内入力の一度だけの複製 | 重み配置の変換、入力ロードのループ・アドレス計算、同一入力の並べ替え、padding tokenの積和 | 対象200 tensorを準備updateで並べ替え、入力は`QuantizedRows`内で共有。主命令2.3933%減、cache容量は同じ。8出力行の分割境界にも対応。[LANE_PAIR.md](LANE_PAIR.md)、[OUTPUT_PAIRS.md](OUTPUT_PAIRS.md) |

最新の採用記録で固定準備は721 owner update・60,994,167,929 handler命令。直前の活性化表版より準備は45,916,168,229命令増えた。「準備1回＋prefix1回＋同じ主問題N回」の命令合計ではN=6からoutput-pairs版が少ない。これは命令合計の比較で、update/queryの料金比較ではない。[OUTPUT_PAIRS.md](OUTPUT_PAIRS.md)。

## 4. 演算を同じqueryへつなぎ、中間処理をなくす

融合では中間配列のencode/decode、checksum、返信・再送に加え、同じ入力の再計算も減る。ただしquery数を減らしても、命令合計が必ず減るとは限らない。

| 融合対象 | 除いた境界処理 | 主suffix87でのquery変化・詳細 |
| --- | --- | --- |
| base＋LoRA A/B＋BF16丸め、出力行tile | 入力の再送、LoRA Aの再計算、中間配列の往復 | 初期の4演算→1 query。各BF16丸め境界は維持。[OPTIMIZATIONS.md](OPTIMIZATIONS.md)、[EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md) |
| 残差＋RMSNorm、MLP gate/up＋SwiGLU＋down | 残差・norm・gate/up・activationの中間返信と再入力 | add/norm融合、MLP/norm融合、prepared-input pipeline、full MLPへ順に拡張。169→138、後のcolumn32版で122→91 query。[BOTTLENECKS.md](BOTTLENECKS.md)、[MLP_NORM_FUSION.md](MLP_NORM_FUSION.md)、[MLP_PIPELINE.md](MLP_PIPELINE.md)、[MLP_FULL.md](MLP_FULL.md)、[COLUMN32.md](COLUMN32.md) |
| 整数dot＋F32 scale/加算 | blockごとの一時I32配列の初期化・store/load | query数は同じ、主命令1.6821%減。元のscale適用順・block加算順を維持。[DOT_SCALE_FUSION.md](DOT_SCALE_FUSION.md) |
| Attention K/V、Q/norm/RoPE/GQA/gate、出力投影 | K/V量子化の重複、正規化・attention結果の返信と再送 | 初期融合305→265、full版138→122 query。[ATTENTION_FUSION.md](ATTENTION_FUSION.md)、[ATTENTION_FULL.md](ATTENTION_FULL.md) |
| Delta投影・gate・conv・再帰・出力投影 | 量子化・LoRA A・gateの再計算、prepared input/gated output/状態の往復 | projected版265→193、finish版193→169、full-log版91→67 query。prefixの元F32更新量logを可逆復元して32 headをまとめる。full-log化は**主命令0.1462%増**、通信38.0105%減。[FUSED_STAGES.md](FUSED_STAGES.md)、[DELTA_PROJECTED.md](DELTA_PROJECTED.md)、[DELTA_FINISH.md](DELTA_FINISH.md)、[DELTA_FULL_LOG.md](DELTA_FULL_LOG.md) |
| byte buffer・metadata・残差の受渡し | Candid byteの要素ごとの復元、内部の二重metadata検証、残差コピー | balanced44との組合せで主命令2.178%減。[BYTE_BUFFER.md](BYTE_BUFFER.md) |

## 5. 判断に不要な計算・重複通信を省く

| 手段 | 省略した対象 | 実測・詳細 |
| --- | --- | --- |
| readoutを候補数＋unknown行に限定 | 使わないreadout出力行、state/logitの重複decode | 初期の23実入力で専用readout命令96.8%減。全モデルの削減率ではない。[OPTIMIZATIONS.md](OPTIMIZATIONS.md) |
| 最終層を必要な最後のtokenに限定 | 最終層の不要なQ、Attention出力、O射影、MLP、残差・norm | K/Vと前31層は全tokenを保持。prefix側の不要なAttention Qも省く。主query379→352、命令3.19%減。[TERMINAL_READOUT.md](TERMINAL_READOUT.md) |
| 終端Delta stateの返送省略・GQA K/V共有 | 継続しない推論の終端state返信、4 query heads間で重複するK/V | prefix準備・継続に必要なstateは保持。導入時の通常実行で通信5.265%減。[COMPACT_HEADS.md](COMPACT_HEADS.md) |
| 可逆BF16/block256 codec | BF16で正確に表せる値を4 byte F32で運ぶ処理 | BF16値は2 byte、F32が必要な値は元bitで保存。大型tileとの組合せで初期通信5.38→2.55 GB。block256版は主通信をさらに5.576%減。[COMMUNICATION.md](COMMUNICATION.md)、[BLOCK_CODEC.md](BLOCK_CODEC.md) |
| frame checksum軽量化・client encode共有 | 毎queryの重複encode/hash、状態frameのSHA-256処理 | version2/BLAKE3 SIMDへ変更し、主handler命令8.4408%減。旧frame互換を維持。[FRAME_CHECKSUM.md](FRAME_CHECKSUM.md) |

## 採用版と追加検証版の到達点

記録上の主canister採用版はoutput-pairs V2、module `36c04a57…`。LoRA入力共有＋MLP89は専用canisterで検証したmodule `3459fc6d…`であり、主canister採用とは分ける。

| 条件 | 主canister採用版：query / handler命令 / Candid byte | LoRA共有＋MLP89検証版：query / handler命令 |
| --- | --- | --- |
| 共通prefix45 tokenの準備 | 66 / 136,863,422,956 / 71,105,691 | 66 / 136,829,434,682 |
| 主問題：全132 token、suffix87 | 67 / 265,931,336,448 / 113,709,005 | 67 / 265,861,866,956 |
| 情報不足：全125 token、suffix80 | 67 / 244,255,114,705 / 106,668,202 | 67 / 244,164,792,824 |
| 重大変更：全134 token、suffix89 | 98 / 275,371,260,925 / 197,070,502 | 67 / 272,246,195,657 |
| prefixなし132 token | 292 / 405,740,371,751 / 580,229,781 | 292 / 405,740,641,584 |

採用版の主問題初回はprefix込み133 query・402,794,759,404命令・184,814,696 Candid byte。固定モデル準備の721 updateはこの推論集計の外にある。主67 queryの内訳はMLP31、Delta24、Attention8、その他4。MLPが命令の53.51%を占める。[OUTPUT_PAIRS.md](OUTPUT_PAIRS.md)、[LORA_MLP89.md](LORA_MLP89.md)。

50/32 query目標は未達。採用版の265.931B handler命令だけでも、測定で用いた5B/queryの予算で最低54回相当となる。50回へは少なくとも約5.99%、32回へは約39.83%の追加命令削減が必要で、実際にはCDK処理と分割境界の余裕も必要となる。

## 試したが採用していない手段

再試行時に同じ探索を繰り返さないため、結果と検証範囲も残す。

| 候補 | 結果・採用しない理由 | 記録 |
| --- | --- | --- |
| 重みscaleの事前復元・検証 | 部分投影では減る形状もあったが、全5条件で命令増。主0.0385%増 | [PREPARED_SCALES.md](PREPARED_SCALES.md) |
| pair補正、Strassenの固定係数・再構成 | 積数が減っても変換・補正・ロード費用が上回る版が多い。二段Strassenは主Q0.2069%増。C64/V4は主Q1.6108%減だが45 token0.7293%増、係数容量3.5倍で全モデル未統合 | [PAIR_FACTOR.md](PAIR_FACTOR.md)、[STRASSEN_PREPACKED.md](STRASSEN_PREPACKED.md)、[STRASSEN_WIDE.md](STRASSEN_WIDE.md)、[STRASSEN_PREPARED.md](STRASSEN_PREPARED.md)、[STRASSEN2.md](STRASSEN2.md)、[STRASSEN16.md](STRASSEN16.md) |
| Winograd二段の共有積・共有加算 | nativeの数値一致まで。文書記録ではWasm性能の実測待ち | [WINOGRAD2.md](WINOGRAD2.md) |
| 全tokenを動的ループで処理し重み展開を共有 | 主Q2.563%増。ループ費用が節約を上回る | [LINEAR_PAIR.md](LINEAR_PAIR.md) |
| 固定重みの安全窓でI16部分和 | 数値一致したが主Q38.9415%増。生成Wasmに符号拡張・乗算・I32復帰・分岐が残る | [BOUNDED_I16.md](BOUNDED_I16.md) |
| 検証済みspanの内部アドレスoverflow確認を省略 | 主Q単体0.4768%減。全モデル比較未完了で主canister未採用 | [BOUNDED_ADDRESS.md](BOUNDED_ADDRESS.md) |
| prefixの一部をdense状態、一部をlogで保持 | 4-bit exponent/SIMD版は復元部分約48.6%減。通信が増え、全推論へ未採用。全モデル削減率とは区別する | [PREFIX_HYBRID.md](PREFIX_HYBRID.md) |
| Relaxed SIMD整数dot | 選択中local環境のWasm validationで拒否。推論・性能実測には進めなかった | [RELAXED_SIMD.md](RELAXED_SIMD.md) |
| LoRAのbase統合、Hadamard回転、量子化粒度の変更 | 元の丸め・量子化契約が変わる精度探索。ホストや射影標本の結果を採用版の全モデル命令削減へ計上しない | [MERGED_ADAPTER_HOST.md](MERGED_ADAPTER_HOST.md)、[EXPLORATION.md](EXPLORATION.md)、[KERNEL_UNROLL.md](KERNEL_UNROLL.md) |

## 数値の読み方と今後の追記

各削減率はそれぞれの直前版・対象入力との比較であり、表の割合を加算しない。部分kernelと全モデル、prefix準備済みと初回、採用版と診断候補も別の測定である。

整数方式導入後の数値保存型の改善は、比較対象の保持hidden/state、必要な最終hidden、logit・確率・判断のbit一致を確認している。これは元の公式BF16モデルとの一致や判断精度の改善を意味しない。重大変更でgold=yes/model=noとなる既存の誤判定は残る。F32超越関数にはnative/Wasmの既存差があるため、同じbackendの前後比較とbackend横断比較を区別する。

handler命令はCDKのCandid decode/encodeを含まず、通信はCandid要求＋返信のみ。単回時間は負荷を統制した反復測定ではなく、命令が減っても時間が増えた条件がある。heap値はhandler終端の観測で、瞬間peakではない。

今後は各手段について「除いた処理」「採用・検証済み未反映・不採用の別」「比較baselineと入力」「全モデル命令/query/通信」「準備・メモリ費用」「数値一致範囲」「詳細記録へのリンク」を追記する。質問依存の中間状態はclient-held、推論は通常query、固定モデル準備のみowner updateという構成を前提にする。
