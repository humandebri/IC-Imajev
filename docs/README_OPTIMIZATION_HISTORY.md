# READMEに掲載していた最適化・測定履歴

2026-10-07にREADMEの冒頭から移した記録。各段階の「最新」「未達」「未接続」は、その記録時点を表す。現在の到達点は[README](../README.md)と[proposalフィルター評価報告](PROPOSAL_FILTER_REPORT_20261007.md)を参照する。入力長、モデル、準備条件が異なる値を一つの性能推移として扱わない。

2026-10-06：都度払いの`infer`を実装。利用者canisterの外部update 1回で結果を返し、内部workerは617/620/653で5/4/5回。各3回のローカル測定で判定・保存中間状態が元モデルとbit一致。前払いcredit口座・timerは使わない。[実測と利用手順](PAID_UPDATE_INFERENCE_MEASURED.md)。

[ランタイムと再利用知見](runtime/README.md)：共通crate、命名、モデルとの境界、llama_cpp_canister比較を整理。

[計算量削減の手段まとめ](COMPUTE_REDUCTION.md)：削減した処理、実測効果、準備費用、採用・未採用の状況を一覧化。

2026-10-03最新：固定INT8配置を準備updateで置換し、query内の入力変換を共有。分割queryの8行境界でも元byteへ読み戻さず、全5条件で保持hidden/state/判断/確率が一致、失敗/replay0。主65.21億handler命令（2.3933%）減、全条件2.39〜7.78%減。主67/初回133 query・通信量は同じ、50/32未達。cache容量は同じ4.065 GB、一度の準備は609.94億命令へ増加。単回時間は主・prefixなし等で悪化し速度改善は未確認。採用module36c04a57…、build時`experimental-prepared-output-pairs`、準備・検証時`--require-output-pairs`を追加。 [実装・全5条件実測](OUTPUT_PAIRS.md)。

2026-10-03追加探索：query内で一度作る入力変換をSIMD化し、64出力で入力ロードを共有する整数Strassenを診断。4版各22通常queryでnative・採用版と出力一致。V4は内部添字の重複overflow確認を省き、主Q投影1.6108%減、45 tokenでは0.7293%増で未採用。全モデルは主67 queryのまま、50/32未達。 [反復処理の削減と実測](STRASSEN16.md)。

2026-10-03最新：毎要素のBF16 SiLU・sigmoid等を固定準備のlookupへ置換し、繰り返すexp・除算・logを除去。全5条件のhidden/state/判断/確率bit一致、失敗/replay0。主75.48億命令（2.6956%）減、全条件2.52〜2.70%減。主67/初回133 query・通信量は同じ、50/32未達。追加固定表1 MiB、採用module ba47cf87…、build時`experimental-prepared-activation`、準備・検証時`--require-prepared-activation`を追加。 [実装・全5条件実測](PREPARED_ACTIVATION.md)。

2026-10-03追加探索：固定INT8重みをK4/2出力の配置へ並べ替え、入力ロードのloopと繰り返すアドレス計算を除去。3版54通常queryでbit一致、最終版の主base Q投影は入力複製込み2.5513%減。初期版25.7%増・明示版4.09%増は不採用。この診断時点では全モデル未接続だった。後続の[固定cache統合と全5条件検証](OUTPUT_PAIRS.md)で、8行分割境界も修正して採用した。主67 query、50/32未達。[実装と測定](LANE_PAIR.md)。

2026-10-03最新：32 head Deltaの投影・conv・再帰・出力投影を1通常queryへ統合。prefixの元F32更新量logをclient-heldで保持し、量子化/LoRA A/gateと固定conv/normを共有。全5条件のhidden/state/判断/確率bit一致、失敗/replay0。主91→67 query、通信183.43→113.71 MB（38.011%減）、初回181→133 query。復元追加で主命令0.146%増、prefixは2.884%減、初回合計0.940%減。prefixなし292 query、50/32未達。採用module8eb3e3b9…、`experimental-delta-full-log`と`--fuse-delta-full-log`を追加。 [実装・全5条件実測](DELTA_FULL_LOG.md)。

2026-10-03最新：INT8入力ロードを32出力で共有し、主87 tokenの全31層MLPを1 queryへ統合。毎回の分割中間状態の返信・復元を除去。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay0。主122→91 query、通信262.96→183.43 MB（30.242%減）、命令2.645%減。初回prefix込み212→181、prefixなし292 query。最大query48.694億命令、50/32未達。対象外条件は合計5,106〜5,880命令増、既存誤判定は残る。採用module e3d8b36a…、`experimental-column32`（`experimental-mlp-full`でも自動有効）。 [実装・全5条件実測](COLUMN32.md)。

2026-10-03追加探索：Delta prefixの元F32更新量を保持する可逆logを実装。144通常queryで全24層・32 headの状態復元bit一致、32 head復元は1.750億命令。統合案の実packet120件は2 MB以内。全モデルへ未接続で採用版は91 queryのまま、50未達。[実装・測定と次の統合](DELTA_INNOVATION_LOG.md)。

2026-10-03最新：INT8重み再展開を３→２回へ減らし、Candid byte復元を一括化、MLP内部の二重metadata検証と残差コピーを省いたv3を採用。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主63.95億命令（2.178%）減、主122/初回212 query・通信量は同じ。対象外条件に最大0.022%の微増。87 token全MLPは実query５B上限を超え不採用、80以下だけ統合を維持。50/32未達、速度改善・一般精度改善は主張しない。採用module bcacce7a…、build時`experimental-byte-buffer`と`experimental-balanced44`（`experimental-mlp-full`でも自動有効）。 [実装・全層比較](BYTE_BUFFER.md)。

2026-10-03最新：K/V投影・Q/GQA・出力投影を１通常queryへ接続し、中間返信と再送を除去。通常層のINT8入力量子化も共有する。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主138→122 query、通信15.99 MB・4.97億命令減。prefix139→123、初回277→245、prefixなし294→292。最大query47.673億命令、50/32未達。単回時間は増減があり速度改善を保証しない。採用module dd0f9dde…、build時`experimental-attention-full`・実行時`--fuse-attention-full`を追加する。 [実装・全層比較](ATTENTION_FULL.md)。

2026-10-03最新：整数dot直後に元のF32 scale/加算を行い、毎blockの一時整数配列の初期化・store/loadを除去。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主50.31億命令（1.6821%）・prefixなし98.63億（2.1566%）減。主138・初回277・prefixなし294 queryと通信量は同じ、50/32未達。最大query39.145億命令。採用module b2ff8580…、build時`experimental-dot-scale`を追加する。 [実装・全層比較](DOT_SCALE_FUSION.md)。

2026-10-03最新：MLPの量子化済みINT8入力・scale・元F32 LoRA A積をクライアントが保持し、次の通常queryが直接利用する２query pipelineを採用。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主169→138 query、通信75.25 MB（21.2446%）・21.43億命令（0.7113%）減。prefix170→139、初回339→277、prefixなし132 tokenは294のまま。最大query39.964億命令、50/32未達。追加量子化なし、既存誤判定は残る。採用module cbd1aac0…、build時`experimental-mlp-pipeline`・実行時`--fuse-mlp-pipeline`を追加する。 [実装・全層比較](MLP_PIPELINE.md)。

2026-10-02最新：元F32 LoRAの端数tokenを16/8/4-groupと独立SIMD laneへまとめ、繰り返す重みロードを削減。回帰する小出力・端数0/4の形状は従来kernelを選択。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主142.45億命令（4.5150%）、prefix41.05億（2.4651%）減。主169・初回339・prefixなし294 queryと通信量は同じ、50/32未達。単回時間は増減があり速度改善を保証しない。採用module2fc01ab1…、build時`experimental-matrix-tail`を追加する。 [実装・全層比較](MATRIX_TAIL.md)。

2026-10-02最新：INT8投影の固定重み展開を48 tokenで共有し、全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主10.55億命令（0.3333%）、prefix11.03億（0.6579%）減。主169・初回339・prefixなし294 queryと通信量は同じ、50/32未達。単回実時間は前回より遅く、速度改善とは扱わない。採用module55e47876…、build時`experimental-column16-token48`へ変更する。 [実装・全層測定](COLUMN16_TOKEN48.md)。

2026-10-02最新：Delta後半のgated出力を通常query内で出力投影へ直接接続し、wrapperの重複した有限値走査を除去。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主193→169 query（12.435%減）、通信17.14 MB（4.6148%）・3.71億命令（0.1171%）減。prefix194→170、初回387→339、prefixなし132 tokenは294のまま。50/32未達。採用module023bda91…、build時`experimental-delta-finish`、実行時`--fuse-delta-finish`を追加する。 [実装・全層実測](DELTA_FINISH.md)。

2026-10-02最新：INT8投影を16出力行で共有し、生成Wasmに残った重み展開loop・一時store/loadを明示macroで除去。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題48.54億命令（1.5084%）、prefix59.79億（3.4419%）減。主193・初回387・prefixなし294 query、通信量は同じで50/32未達。採用module d49fd4e1…、build時`experimental-column16`を追加する。 [実装・全層測定](COLUMN16.md)。

2026-10-02最新：量子化・準備状態復元・返信追加の全域ゼロ初期化→上書きを除去し、実入力は一度だけ書きpaddingだけ初期化。固定RoPE表もowner準備updateで一度生成する。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主問題133,155,010命令（0.04136%）減、prefixなし304,742,867（0.06559%）減。query/通信は同じで主193・初回387・prefixなし294、50/32未達。採用module7fc0e373…、build時`experimental-prepared-rope`、準備時`--require-prepared-rope`を明示する。 [実装・全層測定](ACTIVATION_BUFFERS.md)。

## 2026-10-02の全層推論と最適化

2026-10-02最新：DeltaのQKV/Z projection・gate・conv/再帰を通常queryへ統合し、量子化・元F32の2つのA積・gateをclient-held capture/reuseで一度だけ計算。全5実行で従来Wasmの保持hidden/state/判断/確率bit一致、失敗/replay0。主問題265→193 query、通信459.92→371.33 MB（19.262%減）、命令3234.18→3219.11億（0.466%減）。初回prefix込み507→387、prefixなし438→294 queryで50/32未達。採用modulecfa16f56…、`--fuse-delta-projected`を明示する。既存native/Wasm gate差は分離して確認し、native同士とWasm同士で一致を検証。整数dotの演算方式が次の対象。 [実装・全層比較](DELTA_PROJECTED.md)。

2026-10-02最新：AttentionのK/Vで量子化を共有し、K norm/RoPEとQ projection→norm/RoPE→GQA→gateを通常queryへ統合。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題305→265 query（13.115%減）、通信506.37→459.92 MB（9.173%減）、命令3247.65→3234.18億（0.415%減）。初回prefix込み587→507、prefixなし485→438 queryで50/32未達。単回時間には増加例があり速度改善は主張しない。採用module58cf6426…、`--fuse-attention`を明示する。次のDelta投影統合は5条件の実packetが2MBに収まることだけを確認し、kernelは未実装。 [実装・全層比較](ATTENTION_FUSION.md)。

2026-10-02最新：毎queryで二重に行っていたclient request encode・hashを1回にまとめ、状態frame checksumをversion2/BLAKE3 SIMDへ軽量化。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題3547.05億→3247.65億命令（8.4408%減）、prefixなし5194.94億→4709.36億（9.3472%減）。query数305／初回587／prefixなし485、通信量は同じで50/32未達。固定重み準備はmodel変更/upgrade時のみ、推論は通常query・中間状態はclient-heldを維持。採用module412c565e…、旧frame互換を残し`--frame-checksum blake3`で明示する。 [実装・全層実測](FRAME_CHECKSUM.md)。

2026-10-02最新：分割MLPでもblock256量子化とgate/up両方の元F32 LoRA Aを再利用し、31組の再計算を省いた。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。prefixなし132-tokenは追加31.57億命令（0.6040%）減の5194.94億、485 queryは同じ、通信4.51 MB増。主問題は再利用対象0で305 query・3547億命令、初回prefix込み587 query。50/32未達。採用module752a35b4…、再利用は明示オプション、重み準備はmodel変更/upgrade時だけ。 [実装・全層比較](MLP_REUSE.md)。

2026-10-02追加探索：固定weight scaleの復元・配列確保・正の有限値の再走査を準備updateへ移す候補を実装し、全5実行で保持hidden/state/判断/確率bit一致を確認。ただし主問題は命令0.0385%増、全5実行で増えたため不採用。検証済みソースarchiveと差分を保存し、運用ソース31 hash・主canisterを前の直接INT8展開版e961dda8…へ復元。query数305、初回587、prefixなし485、50/32未達は同じ。F32状態の可逆圧縮も3形式を実測したが、最良でも平均1.83 MB/stateで87-token入力と2 MBへ収まらず、本番codecには未接続。 [候補と全層比較](PREPARED_SCALES.md)。


2026-10-02追加探索：分割間の入力量子化とLoRA Aをclient-held stateで再利用するcapture/reuseを実装。2queryを維持し、2候補・168成功queryで独立scalarとbit一致、不正状態4件を拒否。ただし132-token投影はSIMD版でも命令1.285%増・通信21.44%増で採用保留。整数のまま渡す可逆codecを試作し、次はWasmで検証する。全層305 query・初回587・prefixなし485、50/32未達は変わらない。 [実装・比較](PROJECTION_REUSE.md)。


2026-10-02追加探索：Strassen係数の展開・全要素検証を準備updateへ移し、毎SIMDロードの行アドレス計算も共有した。3候補・合計162通常queryで独立scalarとbit一致したが、87-token base Qは候補16.40億・通常INT8 14.36億命令で不採用。主canisterは既存検証済み版と721 tensor cacheを保持。全層3547億命令・305 query（初回prefix込み587、prefixなし485）と50/32未達は変わらない。 [探索結果](STRASSEN_PREPARED.md)。

2026-10-02最新：元F32 adapter等も準備updateで復元・有限性検証し、通常queryでは同じ配列を共有。主問題3584.2億→3547.0億命令（さらに1.038%減）、全5条件でhidden/state/判断/確率bit一致。stepのstable readは488 MB→0.566 MB、305 query・通信506 MBは同じ。初回prefix込み587 query、prefixなし485 query。重み準備は別途721 update・150.28億命令・234.703秒、cache4.065GB・終端heap最大4.115GB。50/32 query未達。 [実装・全層実測](PREPARED_F32.md)。

2026-10-02前段階：固定INT8 dense重み272個（3.575GB）を準備updateで一度保持し、通常queryは借用して読み出し・バッファ確保・コピーを省く。主問題3654.5億→3584.2億命令（1.923%減）、全5条件でhidden/state/判断/確率bit一致。305 query・通信506 MBは同じ、初回prefix込み587 query。準備は別途272 update、94.36億命令・85.521秒。heap最大観測3.600GB、selected local heap limitは4GiB。50/32 query未達。 [実装・準備手順・全層検証](BORROWED_WEIGHTS.md)。

2026-10-02前段階：量子化済み入力を変更不能な型にし、投影ごとの全走査による再検証を除去。主問題3675億→3655億命令（0.560%減）、305 query・通信506 MBは同じ。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay 0。prefixなし132 tokenは5370億命令（0.742%減）・485 query、初回prefix込み587 query。50/32 query未達。 [実装・実測・次の固定処理削減](QUANTIZED_INPUT_REUSE.md)。

2026-10-02前段階：RoPEの周波数・sin/cosと固定A_logのexpを共有し、Q/K正規化をRoPE queryへ統合。主問題321→305 query、通信519→506 MB、3691億→3675億命令。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。prefix準備298→282 query、初回合計587 query。cacheなし132 tokenは501→485 query。50/32 query未達。 `--fuse-norm-rope`で有効。[実装・全層測定・残る再計算](ROPE_REUSE.md)。

**2026-10-02前段階：同じQ/Kの正規化と同じ入力のactivation量子化を再利用し、主問題3713億→3691億命令へ0.589%減。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。321 query・通信519 MBは同じ、初回prefix込み619 query。cacheなし501 query・5434億命令。50/32 query未達。次はRoPEと固定A_logの再計算を除く。** [実装・全層比較・残る重複処理](DELTA_REUSE.md)。

**前段階（2026-10-02）：可逆block256通信を実装し、主問題の通信550→519 MB（5.576%減）、命令3773億→3713億（1.589%減）。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。主問題321 query、prefix準備298 queryは同じで、50/32 queryは未達。実入力25形状の命令内訳も測定し、整数dotとLoRA積和を主要な削減対象と確認した。** `--wire-codec bf16-block256-exact-v1`で有効。[全層検証・内訳・再現手順](BLOCK_CODEC.md)。

**前段階（2026-10-02）：残差/normをMLPへ統合し、主問題352→321 query、通信579→550 MB。3773億命令（さらに0.576%減）、全5実行で保持hidden/state/判断bit一致、失敗/replay 0。prefix準備は330→298 query、初回合計619 query。cacheなし132 tokenは従来経路の501 query。50/32 queryは未達。** `--fuse-mlp-norm --fuse-mlp --fuse-add-norm`で有効。[全層検証と次の通信試作](MLP_NORM_FUSION.md)。

**前段階（2026-10-02）：整数結果のSIMD取り出しとDelta係数共有を拡大し、主問題3891億→3795億命令へさらに2.48%削減。全5実行と整数126条件がbit一致、失敗/replay 0。352 query・通信579 MB、prefix準備330 queryは同じ。50/32 queryは未達。単回の実時間は増えた例もあり、速度改善とは扱わない。** [全層・実時間・命令構成の検証](PACKED_REDUCTION.md)。

**前段階（2026-10-02）：Delta再帰の係数共有を実装し、主問題の命令数を4053億→3891億へさらに3.99%削減。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。query数352、通信579 MBは同じ。prefix準備は別途330 query、50/32 queryは未達。** [全層検証と残るボトルネック](DELTA_SIMD.md)。INT8事前変換Strassenは87-token Q投影の命令数が約2倍となり不採用。[比較実測](STRASSEN_PREPACKED.md)。

**前段階（2026-10-02）：整数dotの一時配列とDelta状態storeを削減。主問題132 tokens・prefix準備済みは352 queryのまま、4507億→4053億命令（10.06%減）。89-token MLPの統合で最大変更は383→352 query。全5実行で保持hidden/state/判断bit一致・失敗/replay 0。prefix準備は別途330 query。50/32 queryは未達。** [実測と探索](KERNEL_UNROLL.md)。

**前段階の追加改善（2026-10-02）：最終tokenへの演算省略・終端MLP統合・QKV/ゲート射影統合を実装。45-token prefix準備済みの132-token入力は379→352 query、命令4655億→4507億、通信615→579 MB。判断、前31層の全hidden、最終層の最後のhidden、保持stateが従来INT8版とbit一致。prefix準備は別途330 query。32 queryは未達。** [実測と省略範囲](TERMINAL_READOUT.md)。

**前段階の追加改善（2026-10-02）：Delta演算・ゲート射影の統合と短いMLPの分割削減により、共通45-token prefix準備済みの132-token入力は603→379 query、命令4854億→4655億、通信807→615 MB。全32層のhidden・保持state・判断が従来INT8版とbit一致。prefix準備は別途354 query。32 queryは未達。** [実装・検証・残る制約](FUSED_STAGES.md)。

**前段階の追加改善（2026-10-02）：終端Delta状態の返送省略とGQAのK/V共有を実装。通常132-token入力は708→700 query、命令1.05%減・通信5.27%減。共通prefix準備済みでは611→603 query、命令1.46%減・通信7.53%減（807 MB）。準備は別途570 query。3入力で全32層hidden・保持した状態・判断が旧INT8版とbit一致。50 queryは未達。** [実装・測定・再現](COMPACT_HEADS.md)。

**前段階の追加改善（2026-10-02）：INT8直接ロードとtile変更で、cacheなし708 queryの命令数を7393億→7110億へ3.82%削減。共通45-tokenのclient cache利用時は611 query・4926億命令・0.873 GB。ただし初回cache準備は別に578 query・2557億命令・0.464 GB。3入力で全32層hidden/state/判断が以前のINT8版とbit一致。50 queryは未達、速度向上は未確認。** [複数方向の実装・不採用候補・初回費用](DIRECTIONS.md)。

Imajev-4Bのテキスト判断をInternet Computerの通常queryで全32層実行する実験です。INT8 base＋F32 adapter/readoutを固定し、中間状態はclientが保持します。

前段階では整数積和を全モデルへ接続し、BOOM DAO 617で総命令数を当初から72.24%削減した。 132 tokens・rotations=1は1.022兆命令、932 query、Candid通信1.671 GB、単回107.274秒。50 queryは未達です。23件のホスト診断で同じpackのF32演算とラベル一致、元順序の正解付き7問は両方6/7。ただし確率差は最大約8.48ポイントで、一般精度・校正の同等性を保証しません。[整数方式・実測・再現](INTEGER_ARITHMETIC.md)。

数値を変えないF32演算経路も残しています。最新の可逆通信版は1.762兆命令・1,116 query、全層/state/logit/確率が以前の可逆版とbit一致。[追加削減と50-query予算](FIFTY_QUERY_ANALYSIS.md)、全query比較の生データはローカルの`docs/multi-token-compact-lossless-results.json`に保存している（共有資料には含めない）。整数方式は明示した場合に使い、通信は可逆のまま演算差を評価します。

既存INT8通信は可逆BF16通信版から候補確率が最大約0.121変わる別の量子化です。[通信INT8の条件](ACTIVATION_INT8.md)。全モデルの構成・重み準備は [docs/FULL_INFERENCE.md](FULL_INFERENCE.md)。旧最適化の測定履歴は [docs/EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md)、[docs/LAYA_COST_ANALYSIS.md](LAYA_COST_ANALYSIS.md)。

測定と残作業は [docs/STATUS.md](STATUS.md)。採用revision、ファイル容量、SHA256、tensor形状は [MODEL_LOCK.json](../MODEL_LOCK.json)。`PLAN.md` の当初計画を残しています。

追加の整数dot・Delta state store削減と89-token MLP統合は [docs/KERNEL_UNROLL.md](KERNEL_UNROLL.md) を参照。主問題の命令は9.07%減、最大変更は383→352 query。目標50/32 queryは未達。
