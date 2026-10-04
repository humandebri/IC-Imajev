2026-10-04: 実装レビューを修正し、層境界の連結経路を全体へ接続。主87/情報不足80は62→54 query・全出力ビット一致。89 tokenは入口超過を保存し標準62queryへ切替。主の総命令3.11%・通信9.12%増のため実験オプションで、50/32未達。全6条件回帰・最終3条件と54query再開を確認。[実測と修正](docs/ROLL_QUERY.md)。

2026-10-04: 部分down前倒しを追加。87 token/1600行と89 token/1280行でcarry/finish一致、89 token/1600行は5B超過。全6条件回帰も通過。全体graphは未接続で主62 query、50未達。[実測と次のボトルネック](docs/MLP_DELTA_STREAM.md)。

2026-10-04: MLP完了→次層Delta部分→次queryのDelta完了/次層MLP準備を実装。39成立条件でcarry/次層finishまでビット一致、45条件は5B超過を記録。新Wasmの全6条件回帰も通過。全体は未接続で主62 query、50未達。詳細は [MLP_DELTA_STREAM.md](docs/MLP_DELTA_STREAM.md)。

2026-10-04: レビュー修正を `597206a` にコミット。続いてMLP最終chunkとfinishのquery融合を実装し、45実測条件でhidden/normビット一致、分割比1 query・約92〜105M命令・約2.86〜3.18MB通信を削減。全体graphへは未接続で、主62 query。詳細は [MLP_COMPLETE.md](docs/MLP_COMPLETE.md)。

# Imajev-4Bのローカルcanister実装計画

2026-10-04レビュー：compact tailの古いlayer30出力を除去し、MLP carry送信時の検査用全復元をRust/Pythonとも省略、frame versionの型検査を修正。候補/通常featureのRust・Python検証、保存payload315個のbyte一致、Wasmビルドを確認。実canister命令の再測定は後続に分離。主62 query・50未達を維持。 [レビュー記録](docs/IMPLEMENTATION_REVIEW_2026_10_04.md)。

2026-10-04追加：列継続のS1係数変換を新しい列だけへ限定し、余分なゼロ埋めも除去。実32条件でbit一致・命令減・通信不変、主87の2分割225万/4分割675万命令減。Delta16+6+10 headの実12条件でgated/conv履歴が一致、profileで入力量子化/Aは最初だけと確認。全6条件のgraph回帰も一致。ただし新APIは現graphへ未接続で主62 query、50/32未達。 [実装と実測](docs/INT8_COLUMN_CONTINUATION.md)。

2026-10-03追加：Delta out projectionを列単位で継続し、新しい列だけをINT8量子化・base/F32 A加算、finishではF32 Bだけを実行。実8入力×4分割の32条件で一括出力とbit一致、全6条件のgraph回帰も一致。現graphは未接続で主62 queryのまま、分岐追加で1,241,192命令増。単独分割はquery/通信増のため性能改善として未採用、50/32未達。 [実装と実測](docs/INT8_COLUMN_CONTINUATION.md)。

2026-10-03追加：入力量子化・gate/up F32 Aを一度だけ計算し、product量子化とdown F32 Aの部分和を保持するMLP分割APIを実装。80/87/89 token・実3層・5分割の45条件で準備状態/hidden/normが既存INT8版とbit一致、profileで再実行なしを確認。既存graphは未接続で主62 queryのまま、API分岐追加により8,528命令増。単独分割は通信/命令増のため性能改善として未採用、次はDelta分割との融合。50/32未達。 [実装と実測](docs/MLP_STREAM.md)。

2026-10-03追加：分割MLPに残っていた固定F32重みの元配置復元と入力転置を除去。cold132の全推論で1,447,149,192命令（0.400601%）減、全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致。主87は62 query・命令・通信とも同じで50/32未達。F32列部分和をクライアント保持し加算順序を保って継続する部品も、実3層・251診断queryで一致を検証。 [分割MLPの実測](docs/MLP_PREPARED_CAPTURE.md)、[F32継続演算](docs/F32_COLUMN_CONTINUATION.md)。

2026-10-03追加：整数S1の出力tileを32→128行へ広げ、同じ入力ロードとtoken/scale/address処理を共有。主さらに7,888,349,165命令（3.245%）減、235,201,564,012命令。全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致、失敗/replay0。最大変更89は63→62query、主は62で50/32未達。主単回時間20.221→35.584秒の悪化を記録。256行版はlocals上限でinstall拒否、64/128診断は全11入力一致。 [実装と実測](docs/S1_OUTPUT_TILE_REUSE.md)。

2026-10-03追加：各queryの入力・出力frameハッシュを、署名検証するクライアントへ移管。主さらに2,021,749,972命令（0.825%）減、243,089,913,177命令。全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致、失敗/replay0。62 query・通信は同じで50/32未達。署名改ざん拒否・保存状態破損の送信前拒否・不正payload48件拒否を検証。89-token tail統合は5B超過で分割維持。 [実装と実測](docs/HOST_FRAME_CHECKSUM.md)。

2026-10-03追加：block codecの有限値検査とBF16分類を一走査にし、復号のゼロ埋め・再走査を除去。主さらに788,566,976命令（0.321%）減、245,111,663,149命令。全条件の返却hidden/保持state/判断/確率bit一致、失敗/replay0。62 query・通信・観測heap終了最大は同じで50/32未達。100 codec診断と正しいchecksum付き不正payload32件の拒否も検証。[実装と実測](docs/CODEC_STREAMING_CHECK.md)。

2026-10-03追加：量子化で入力の有限値検査とblock256の最大値探索を一走査へ統合。主さらに1,233,003,258命令（0.499%）減、245,900,230,125命令。全条件の返却hidden/保持state/判断/確率bit一致、失敗/replay0。62 query・通信・観測heap終了最大は同じで50/32未達。単体36通常queryで整数/scale一致とInf/NaN拒否も確認。[実装と実測](docs/QUANTIZE_PEAK_SCAN.md)。

2026-10-03追加：最終2層で検査済み入力を型で引き継ぎ、接続時の全走査・連結コピーと不要なlayer30 hiddenの返送を除去。主63→62 query、51,181,409命令・Candid1,342,937 bytes減、247,133,233,383命令。全条件の返却hidden/保持state/判断/確率bit一致、graph失敗/replay0。layer30 hidden非返却を明示、87以下に限定、89/132は分割経路。50/32未達。[実装と実測](docs/TERMINAL_TAIL.md)。

2026-10-03追加：S1の7係数で繰り返す共通byte offsetをtoken pairごとに一度計算し再利用。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主632,813,440命令（0.25535%）減、247,184,414,792命令。63 query・通信・観測heap終了最大は同じで50/32未達。固定pointer再利用も単体で出力一致だが1token増加により未接続。[実装と実測](docs/S1_ADDRESS_REUSE.md)。

2026-10-03追加：LoRA Aのinput転置を省き、元容量の固定output32配置とK64部分和で全tokenのweight loadを共有。全5条件の保持hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに24.683億命令（0.9834%）減、248,528,396,096命令。64 query・通信は同じで50/32未達。主MLPを再profileし、base77.58%、A合計4.61%、B7.42%を実測。[F32_A_REUSE.md](docs/F32_A_REUSE.md)。

2026-10-03追加：S1入力係数の全領域ゼロ埋め直後の全上書きを除去し、一度だけ書き込む。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主353,828,218命令（0.14257%）減、247,817,228,232命令。63 query・通信・観測heap終了最大は同じで50/32未達。S2共通変換再利用3候補は現行S1より増加し未採用。[実装と実測](docs/OPERAND_WRITE_ONCE.md)。

2026-10-03追加：最終Attention/MLP/normと専用F32判断を1通常queryへ統合し、最後のhidden再送を除去。主64→63 query、495,771命令・Candid10,619 bytes減。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。同じmoduleの2→1呼び出しcontrolとqueryなしcheckpoint再開も検証。固定payload・観測heap終了最大を維持。50/32未達。[実装と実測](docs/TERMINAL_DECISION.md)。


2026-10-03追加：INT8固定scaleのqueryごとの読み出し・byte→F32復号・正値/有限値再走査を除去。固定payload不変、全5条件の保持hidden/state/判断bit一致、失敗/replay0。主356,843,875命令（0.1436%）減、248,171,552,221命令。64 queryと通信は同じで50/32未達。観測heap終了最大は+2,424,832 bytes。第二候補32a3e1d5…を実験用基準とする。[固定scaleの実装と実測](docs/FIXED_SCALE_REUSE.md)。


2026-10-03追加：固定F32 LoRA Bを元容量のoutput32配置にし、全tokenでweight loadを共有、query input転置を除去。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに46.333億命令（1.8125%）減、250,996,660,791命令。cache4.065 GBを維持、64 query・通信は同じで50/32未達。LoRA AのK64診断も主18.738%減・22測定queryでdigest一致し、次の全モデル接続候補とする。[F32_WEIGHT_REUSE.md](docs/F32_WEIGHT_REUSE.md)。

2026-10-03追加：元容量のINT8四象限配置を一度準備し、query内のS1入力変換・初回weight展開を共有。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主50.717億命令（1.9454%）減、全条件0.804〜2.569%減。固定cache容量4.065 GBを維持。主64 query / Candid124,634,637 bytesで50/32未達。詳細と再現：[STRASSEN_RAW_REUSE.md](docs/STRASSEN_RAW_REUSE.md)。

2026-10-03最新：固定INT8配置を準備updateで置換し、query内の入力変換を共有。分割queryの8行境界でも元byteへ読み戻さず、全5条件で保持hidden/state/判断/確率が一致、失敗/replay0。主65.21億handler命令（2.3933%）減、全条件2.39〜7.78%減。主67/初回133 query・通信量は同じ、50/32未達。cache容量は同じ4.065 GB、一度の準備は609.94億命令へ増加。単回時間は主・prefixなし等で悪化し速度改善は未確認。採用module36c04a57…、build時`experimental-prepared-output-pairs`、準備・検証時`--require-output-pairs`を追加。 [実装・全5条件実測](docs/OUTPUT_PAIRS.md)。

2026-10-03追加探索：query内で一度作る入力変換をSIMD化し、64出力で入力ロードを共有する整数Strassenを診断。4版各22通常queryでnative・採用版と出力一致。V4は内部添字の重複overflow確認を省き、主Q投影1.6108%減、45 tokenでは0.7293%増で未採用。全モデルは主67 queryのまま、50/32未達。 [反復処理の削減と実測](docs/STRASSEN16.md)。

2026-10-03最新：毎要素のBF16 SiLU・sigmoid等を固定準備のlookupへ置換し、繰り返すexp・除算・logを除去。全5条件のhidden/state/判断/確率bit一致、失敗/replay0。主75.48億命令（2.6956%）減、全条件2.52〜2.70%減。主67/初回133 query・通信量は同じ、50/32未達。追加固定表1 MiB、採用module ba47cf87…、build時`experimental-prepared-activation`、準備・検証時`--require-prepared-activation`を追加。 [実装・全5条件実測](docs/PREPARED_ACTIVATION.md)。

2026-10-03追加探索：固定INT8重みをK4/2出力の配置へ並べ替え、入力ロードのloopと繰り返すアドレス計算を除去。3版54通常queryでbit一致、最終版の主base Q投影は入力複製込み2.5513%減。初期版25.7%増・明示版4.09%増は不採用。この診断時点では全モデル未接続だった。後続の[固定cache統合と全5条件検証](docs/OUTPUT_PAIRS.md)で、8行分割境界も修正して採用した。主67 query、50/32未達。[実装と測定](docs/LANE_PAIR.md)。

2026-10-03最新：32 head Deltaの投影・conv・再帰・出力投影を1通常queryへ統合。prefixの元F32更新量logをclient-heldで保持し、量子化/LoRA A/gateと固定conv/normを共有。全5条件のhidden/state/判断/確率bit一致、失敗/replay0。主91→67 query、通信183.43→113.71 MB（38.011%減）、初回181→133 query。復元追加で主命令0.146%増、prefixは2.884%減、初回合計0.940%減。prefixなし292 query、50/32未達。採用module8eb3e3b9…、`experimental-delta-full-log`と`--fuse-delta-full-log`を追加。 [実装・全5条件実測](docs/DELTA_FULL_LOG.md)。

2026-10-03最新：INT8入力ロードを32出力で共有し、主87 tokenの全31層MLPを1 queryへ統合。毎回の分割中間状態の返信・復元を除去。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay0。主122→91 query、通信262.96→183.43 MB（30.242%減）、命令2.645%減。初回prefix込み212→181、prefixなし292 query。最大query48.694億命令、50/32未達。対象外条件は合計5,106〜5,880命令増、既存誤判定は残る。採用module e3d8b36a…、`experimental-column32`（`experimental-mlp-full`でも自動有効）。 [実装・全5条件実測](docs/COLUMN32.md)。

2026-10-03追加探索：Delta prefixの元F32更新量を保持する可逆logを実装。144通常queryで全24層・32 headの状態復元bit一致、32 head復元は1.750億命令。統合案の実packet120件は2 MB以内。全モデルへ未接続で採用版は91 queryのまま、50未達。[実装・測定と次の統合](docs/DELTA_INNOVATION_LOG.md)。

2026-10-03最新：INT8重み再展開を３→２回へ減らし、Candid byte復元を一括化、MLP内部の二重metadata検証と残差コピーを省いたv3を採用。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主63.95億命令（2.178%）減、主122/初回212 query・通信量は同じ。対象外条件に最大0.022%の微増。87 token全MLPは実query５B上限を超え不採用、80以下だけ統合を維持。50/32未達、速度改善・一般精度改善は主張しない。採用module bcacce7a…、build時`experimental-byte-buffer`と`experimental-balanced44`（`experimental-mlp-full`でも自動有効）。 [実装・全層比較](docs/BYTE_BUFFER.md)。

2026-10-03最新：K/V投影・Q/GQA・出力投影を１通常queryへ接続し、中間返信と再送を除去。通常層のINT8入力量子化も共有する。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主138→122 query、通信15.99 MB・4.97億命令減。prefix139→123、初回277→245、prefixなし294→292。最大query47.673億命令、50/32未達。単回時間は増減があり速度改善を保証しない。採用module dd0f9dde…、build時`experimental-attention-full`・実行時`--fuse-attention-full`を追加する。 [実装・全層比較](docs/ATTENTION_FULL.md)。

2026-10-03最新：整数dot直後に元のF32 scale/加算を行い、毎blockの一時整数配列の初期化・store/loadを除去。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主50.31億命令（1.6821%）・prefixなし98.63億（2.1566%）減。主138・初回277・prefixなし294 queryと通信量は同じ、50/32未達。最大query39.145億命令。採用module b2ff8580…、build時`experimental-dot-scale`を追加する。 [実装・全層比較](docs/DOT_SCALE_FUSION.md)。

2026-10-03最新：MLPの量子化済みINT8入力・scale・元F32 LoRA A積をクライアントが保持し、次の通常queryが直接利用する２query pipelineを採用。全５条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主169→138 query、通信75.25 MB（21.2446%）・21.43億命令（0.7113%）減。prefix170→139、初回339→277、prefixなし132 tokenは294のまま。最大query39.964億命令、50/32未達。追加量子化なし、既存誤判定は残る。採用module cbd1aac0…、build時`experimental-mlp-pipeline`・実行時`--fuse-mlp-pipeline`を追加する。 [実装・全層比較](docs/MLP_PIPELINE.md)。

2026-10-02最新：元F32 LoRAの端数tokenを16/8/4-groupと独立SIMD laneへまとめ、繰り返す重みロードを削減。回帰する小出力・端数0/4の形状は従来kernelを選択。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主142.45億命令（4.5150%）、prefix41.05億（2.4651%）減。主169・初回339・prefixなし294 queryと通信量は同じ、50/32未達。単回時間は増減があり速度改善を保証しない。採用module2fc01ab1…、build時`experimental-matrix-tail`を追加する。 [実装・全層比較](docs/MATRIX_TAIL.md)。

2026-10-02最新：INT8投影の固定重み展開を48 tokenで共有し、全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主10.55億命令（0.3333%）、prefix11.03億（0.6579%）減。主169・初回339・prefixなし294 queryと通信量は同じ、50/32未達。単回実時間は前回より遅く、速度改善とは扱わない。採用module55e47876…、build時`experimental-column16-token48`へ変更する。 [実装・全層測定](docs/COLUMN16_TOKEN48.md)。

2026-10-02最新：Delta後半のgated出力を通常query内で出力投影へ直接接続し、wrapperの重複した有限値走査を除去。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主193→169 query（12.435%減）、通信17.14 MB（4.6148%）・3.71億命令（0.1171%）減。prefix194→170、初回387→339、prefixなし132 tokenは294のまま。50/32未達。採用module023bda91…、build時`experimental-delta-finish`、実行時`--fuse-delta-finish`を追加する。 [実装・全層実測](docs/DELTA_FINISH.md)。

2026-10-02最新：INT8投影を16出力行で共有し、生成Wasmに残った重み展開loop・一時store/loadを明示macroで除去。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題48.54億命令（1.5084%）、prefix59.79億（3.4419%）減。主193・初回387・prefixなし294 query、通信量は同じで50/32未達。採用module d49fd4e1…、build時`experimental-column16`を追加する。 [実装・全層測定](docs/COLUMN16.md)。

2026-10-02最新：量子化・準備状態復元・返信追加の全域ゼロ初期化→上書きを除去し、実入力は一度だけ書きpaddingだけ初期化。固定RoPE表もowner準備updateで一度生成する。全5条件の保持hidden/state/判断/確率bit一致、失敗/replay0。主問題133,155,010命令（0.04136%）減、prefixなし304,742,867（0.06559%）減。query/通信は同じで主193・初回387・prefixなし294、50/32未達。採用module7fc0e373…、build時`experimental-prepared-rope`、準備時`--require-prepared-rope`を明示する。 [実装・全層測定](docs/ACTIVATION_BUFFERS.md)。

# Imajev-4BをInternet Computerで動かすための実験計画

2026-10-02最新：DeltaのQKV/Z projection・gate・conv/再帰を通常queryへ統合し、量子化・元F32の2つのA積・gateをclient-held capture/reuseで一度だけ計算。全5実行で従来Wasmの保持hidden/state/判断/確率bit一致、失敗/replay0。主問題265→193 query、通信459.92→371.33 MB（19.262%減）、命令3234.18→3219.11億（0.466%減）。初回prefix込み507→387、prefixなし438→294 queryで50/32未達。採用modulecfa16f56…、`--fuse-delta-projected`を明示する。既存native/Wasm gate差は分離して確認し、native同士とWasm同士で一致を検証。整数dotの演算方式が次の対象。 [実装・全層比較](docs/DELTA_PROJECTED.md)。

2026-10-02最新：AttentionのK/Vで量子化を共有し、K norm/RoPEとQ projection→norm/RoPE→GQA→gateを通常queryへ統合。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題305→265 query（13.115%減）、通信506.37→459.92 MB（9.173%減）、命令3247.65→3234.18億（0.415%減）。初回prefix込み587→507、prefixなし485→438 queryで50/32未達。単回時間には増加例があり速度改善は主張しない。採用module58cf6426…、`--fuse-attention`を明示する。次のDelta投影統合は5条件の実packetが2MBに収まることだけを確認し、kernelは未実装。 [実装・全層比較](docs/ATTENTION_FUSION.md)。

2026-10-02最新：毎queryで二重に行っていたclient request encode・hashを1回にまとめ、状態frame checksumをversion2/BLAKE3 SIMDへ軽量化。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題3547.05億→3247.65億命令（8.4408%減）、prefixなし5194.94億→4709.36億（9.3472%減）。query数305／初回587／prefixなし485、通信量は同じで50/32未達。固定重み準備はmodel変更/upgrade時のみ、推論は通常query・中間状態はclient-heldを維持。採用module412c565e…、旧frame互換を残し`--frame-checksum blake3`で明示する。 [実装・全層実測](docs/FRAME_CHECKSUM.md)。

2026-10-02最新：分割MLPでもblock256量子化とgate/up両方の元F32 LoRA Aを再利用し、31組の再計算を省いた。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。prefixなし132-tokenは追加31.57億命令（0.6040%）減の5194.94億、485 queryは同じ、通信4.51 MB増。主問題は再利用対象0で305 query・3547億命令、初回prefix込み587 query。50/32未達。採用module752a35b4…、再利用は明示オプション、重み準備はmodel変更/upgrade時だけ。 [実装・全層比較](docs/MLP_REUSE.md)。

2026-10-02追加探索：固定weight scaleの復元・配列確保・正の有限値の再走査を準備updateへ移す候補を実装し、全5実行で保持hidden/state/判断/確率bit一致を確認。ただし主問題は命令0.0385%増、全5実行で増えたため不採用。検証済みソースarchiveと差分を保存し、運用ソース31 hash・主canisterを前の直接INT8展開版e961dda8…へ復元。query数305、初回587、prefixなし485、50/32未達は同じ。F32状態の可逆圧縮も3形式を実測したが、最良でも平均1.83 MB/stateで87-token入力と2 MBへ収まらず、本番codecには未接続。 [候補と全層比較](docs/PREPARED_SCALES.md)。


2026-10-02追加探索：分割間の入力量子化とLoRA Aをclient-held stateで再利用するcapture/reuseを実装。2queryを維持し、2候補・168成功queryで独立scalarとbit一致、不正状態4件を拒否。ただし132-token投影はSIMD版でも命令1.285%増・通信21.44%増で採用保留。整数のまま渡す可逆codecを試作し、次はWasmで検証する。全層305 query・初回587・prefixなし485、50/32未達は変わらない。 [実装・比較](docs/PROJECTION_REUSE.md)。


2026-10-02追加探索：Strassen係数の展開・全要素検証を準備updateへ移し、毎SIMDロードの行アドレス計算も共有した。3候補・合計162通常queryで独立scalarとbit一致したが、87-token base Qは候補16.40億・通常INT8 14.36億命令で不採用。主canisterは既存検証済み版と721 tensor cacheを保持。全層3547億命令・305 query（初回prefix込み587、prefixなし485）と50/32未達は変わらない。 [探索結果](docs/STRASSEN_PREPARED.md)。

2026-10-02最新：元F32 adapter等も準備updateで復元・有限性検証し、通常queryでは同じ配列を共有。主問題3584.2億→3547.0億命令（さらに1.038%減）、全5条件でhidden/state/判断/確率bit一致。stepのstable readは488 MB→0.566 MB、305 query・通信506 MBは同じ。初回prefix込み587 query、prefixなし485 query。重み準備は別途721 update・150.28億命令・234.703秒、cache4.065GB・終端heap最大4.115GB。50/32 query未達。 docs/PREPARED_F32.mdに記録。

2026-10-02前段階：固定INT8 dense重み272個（3.575GB）を準備updateで一度保持し、通常queryは借用して読み出し・バッファ確保・コピーを省く。主問題3654.5億→3584.2億命令（1.923%減）、全5条件でhidden/state/判断/確率bit一致。305 query・通信506 MBは同じ、初回prefix込み587 query。準備は別途272 update、94.36億命令・85.521秒。heap最大観測3.600GB、selected local heap limitは4GiB。50/32 query未達。 docs/BORROWED_WEIGHTS.mdに記録。

2026-10-02前段階：量子化済み入力を変更不能な型にし、投影ごとの全走査による再検証を除去。主問題3675億→3655億命令（0.560%減）、305 query・通信506 MBは同じ。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay 0。prefixなし132 tokenは5370億命令（0.742%減）・485 query、初回prefix込み587 query。50/32 query未達。 docs/QUANTIZED_INPUT_REUSE.mdに記録。

2026-10-02前段階：RoPEの周波数・sin/cosと固定A_logのexpを共有し、Q/K正規化をRoPE queryへ統合。主問題321→305 query、通信519→506 MB、3691億→3675億命令。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。prefix準備298→282 query、初回合計587 query。cacheなし132 tokenは501→485 query。50/32 query未達。 docs/ROPE_REUSE.mdに記録。固定adapterの準備時統合はホスト23条件で判断一致、元順序gold6/7→6/7だが確率差最大7.64ポイント。canister未採用。

2026-10-02前段階：同じQ/Kの正規化と同じ入力のactivation量子化を再利用し、主問題3713億→3691億命令へ0.589%減。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。321 query・通信519 MBは同じ、初回prefix込み619 query。cacheなし501 query・5434億命令。50/32 query未達。次はRoPEと固定A_logの再計算を除く。 docs/DELTA_REUSE.mdに全実測を記録。Strassenのquery内補正除去は通常積和より遅く不採用（docs/STRASSEN_WIDE.md）。

2026-10-02前段階：可逆block256通信を実装し、新prefixから全5条件を検証。全hidden/state/判断bit一致、失敗/replay 0。主問題321 query、通信519,228,106 bytes、371,307,078,099命令。prefix準備298 query、cacheなし501 query。50/32 query未達。実入力25形状で命令内訳も測定し、整数dotとLoRA積和を主要な削減対象と確認した。通常版へ復帰済み。次は残差/normと次層投影の統合、Strassenの補正費用を省く方式を探索する。docs/BLOCK_CODEC.mdに結果と未実装範囲を記録した。

2026-10-02前段階：残差/normをMLP queryへ統合し、主問題352→321 query、通信579→550 MB、3773億命令。prefix準備330→298 query、初回合計619 query。cacheなし132 tokenはサイズ上限のため従来経路501 queryを保存した。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。50/32 queryは未達。docs/MLP_NORM_FUSION.mdに記録した。次の統合へ、可逆block codecでQKV/gates/hiddenの実payloadを約1.89 MBに収めるpacket試作を行ったが、canisterと拡張要素数の検証は未実装。

2026-10-02最新：整数結果のSIMD取り出しとDelta係数共有の拡大を採用。主問題3891億→3795億命令（さらに2.48%減）。全5実行と追加整数126条件でbit一致、失敗/replay 0。352 query、prefix準備330 queryは同じで50/32 query未達。命令数だけでも50回へさらに約34.1%削減が必要。単回時間の増加も記録し、命令削減を速度向上と混同しない。docs/PACKED_REDUCTION.mdに全実測・生成Wasmの比較を記録した。

2026-10-02最新：Delta再帰の係数共有を採用。主問題4053億→3891億命令、全5実行で保持hidden/state/判断bit一致、失敗/replay 0。352 query、prefix準備330 queryは変わらず50/32 query未達。整数射影/MLPが全体の約79.8%で最大の削減対象。事前変換Strassenは実INT8 Q投影の命令数が約2倍となり不採用。docs/DELTA_SIMD.md、docs/STRASSEN_PREPACKED.mdに記録した。さらに大きなDelta係数共有の候補は生成したが未測定。

2026-10-02追記：整数dotの一時配列とDelta状態store削減を全層検証。主問題352 query・4098億命令、最大変更は89-token MLP統合で383→352 query・4238億命令。新prefixを含む全5実行で保持hidden/state/判断bit一致、失敗/replay 0。50/32 query目標は未達。現在の実装・測定はdocs/KERNEL_UNROLL.md。

2026-10-02追加：token tile32/16/8、出力tile8、INT8直接ロードを比較。改善した出力tile8＋直接ロードを採用し、通常708 queryを0.711兆命令へ削減。共通prefix45 tokenのclient-held conv/DeltaNet/KV state cacheを実装し、準備済み617は611 query・0.493兆命令。初回準備578 query・0.256兆命令を別計上。3入力の全32層/state/判断bit一致。50 query未達。最新はdocs/DIRECTIONS.md。


2026-10-02追加：整数dot256列の正確なI32加算、gate/up/SwiGLU融合とwork予算に基づく分割を実装。通常query全32層で708 query、0.739兆命令、1.248 GB、単回78.490秒。以前のINT8方式と全層/state/判断bit一致。最大45.80億handler命令の設定は明示した実験オプションで、既定の保守的分割は残す。50 queryは未達。最新はdocs/MLP_FUSION.md。


2026-10-01追加：query内stageまで計測し、整数scale/reduction・BF16 codec・RNE量子化をSIMD化、token分割を8境界へ整列、残差/normを融合。0.847兆命令・868 query・全層/state/判断bit保存。50 query未達。最新はdocs/BOTTLENECKS.md。

作成日・更新日: 2026-10-01（日本時間）
状態: 固定モデル・公式ホスト比較・Rust/Wasm・INT8 packの全32層local query推論まで完走。2026-10-01、数値維持の追加改善で1.762兆命令・1,116 query、演算方式変更の許可後は整数base＋F32 LoRAで1.022兆命令・932 queryへ削減。ホスト23件の同じpackのF32/整数ラベル一致を確認したが、確率は変わり、校正維持や一般精度を保証しない。50 queryは未達。最新実測・精度評価はdocs/INTEGER_ARITHMETIC.md。以下の当初計画を残す。mainnet/push/PRは未実施。

## 目的と最初の到達点

Layaより大きい公開の判定モデルを、Internet Computerのcanisterで分割queryとして実行できるか検証する。採用候補はImajev-4Bに決定した。まずテキスト入力だけを対象にする。画像の認識はモデルの能力として存在するが、初回の移植範囲には含めない。2B版は4Bの移植が容量や実行環境で難しい場合の後続候補とし、最初から2Bの移植を必須にしない。

最初の到達点は、短いテキストと2〜7個の選択肢、1質問を入力し、固定した公式実装と比較可能なlogits・確率・unknownの扱いを、ローカルcanisterのqueryだけで返すことである。公式評価に合わせ、1回のforwardに相当する計算を複数queryへ分割し、選択肢順の4回rotation平均は初回には使わない。モデルのuploadと読み込み準備にはupdateを使う。推論途中の状態はクライアントが保持する。

BOOM DAO提案の評価で改善を示せることと、canister上で実行できることを別々の達成条件にする。モデルを大きくしただけで判定精度が上がるとは仮定しない。

## 採用モデルと評価条件

Imajev-4Bを選ぶ根拠は、未公開問題を含むJevBench v1.4.2.2の総合評価で67.37、1位となった比較結果である。総合点は判断能力・校正・速度・費用を組み合わせた指標であり、正解率でも、あらゆる業務で最も優秀という証明でもない。画像対応を採用理由の中心には置かない。

| 対象 | 位置付け | 固定・確認する内容 |
|---|---|---|
| mohit67890/imajev-4b | 採用候補 | 評価済みrevision c9e5f132から開始。base、LoRA、専用readout、校正を一組で固定 |
| Qwen/Qwen3.5-4B | 基盤モデル | 採用adapterと対応するbase revision、config、tokenizer、processor |
| mohit67890/imajev | 参照実装 | 評価済みserver commit a0134749e0900189c129cd6bb5000969f3b64bb5を起点に確認 |
| convaiinnovations/laya-typed-decisions | 比較基準 | IC-Laya側の固定モデル・Wasm・既存結果 |
| JevK5-4B / Plumb-4B | 必要時の比較候補 | ホスト上で同じ問題を評価。移植対象はImajevに絞る |

公式評価の条件はrotations=1とcalibration.jsonである。移動するmainや4回rotationの推奨例を、この条件と混ぜない。実験開始時にadapter revisionの短縮表記を完全なcommitへ解決し、ファイルのSHA256も保存する。必要な参照環境が再現できなければ、その差分を明記する。

ImajevはTypeSafeのJevとは独立した公開モデルであり、Qwen3.5の言語モデルへLoRAと専用decision readoutを加えた構成である。画像も入力できるが、初回はテキストだけを扱う。公式APIのunknown_probabilityとabstainedを含め、情報不足をどう返すかも参照する。判定の確率や棄権は正しさを保証しない。

LocalLLaMA/typed-decisionsを補助ベンチに使う場合は、trainで学習・調整したモデルとzero-shotのモデルを分ける。このデータのgoldは教師モデルの回答分布であり、スコアを実世界の正解率として扱わない。今回の採用はそのベンチへの適応点だけで決めていない。

- Imajev公式実装: https://github.com/mohit67890/imajev
- 4Bモデルカード: https://huggingface.co/mohit67890/imajev-4b
- 固定評価の説明: https://github.com/fstandhartinger/jevbench/blob/main/docs/RELEASE-v1.4.2.2.md
- Layaモデルカード: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Typed Decisions: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- ICリソース上限: https://docs.internetcomputer.org/references/resource-limits/
- IC料金表: https://docs.internetcomputer.org/references/cycle-costs/

## 独立した作業領域

作業領域は`/Volumes/KINGSTON/ICP/IC-Imajev`とし、リポジトリ名は`IC-Imajev`にする。既存のIC-Laya-Standaloneのソース、Git状態、モデル、稼働中のローカルcanisterを変更しない。既存実装は参照元として読み、再利用するコードは依存関係とライセンスを記録して新しいリポジトリへ取り込む。

実装開始後の配置案:

```text
PLAN.md                     この計画書
README.md                   再現可能な実行手順
Cargo.toml                  Wasm向け実行エンジンのworkspace
crates/imajev-runtime/        演算・モデルpack・分割状態
canisters/inference/        upload・準備・query API
client/                     tokenizer・query実行・途中保存
scripts/                    export・変換・比較・測定
fixtures/                   小さな合成tensorと固定入力
benchmarks/                 問題定義・評価手順・正解根拠
checkpoints/                重み・変換pack（Git対象外）
artifacts/                  生の測定結果（Git対象外）
docs/                       設計判断・要約・採用理由
```

Gitを初期化するときに重み、生成pack、ビルド成果物、キャッシュ、秘密鍵、生の測定ログをignoreする。再現用の小さなfixtureと結果要約だけを明示的に管理する。

## 段階1: モデルと実行条件を固定する

重みを取得する前に、固定revisionのbase config、adapter_config、safetensorsのヘッダー、公式のprompt生成と専用readoutコードを調べる。必要な演算、層数、hidden幅、語彙数、tensor配置、LoRAの対象とrank、readout、unknownと校正の設定を記録する。モデルカードの現在のmainと評価済みrevisionの構成を混ぜない。

Qwen3.5の演算を通常のattentionだけと仮定しない。対象checkpointでfull attentionと線形attention・recurrent系ブロックがどう構成されるか確認し、convolution、gating、正規化、位置表現などの必要演算を一覧にする。既存のCandleやCPU実装が対象revisionに対応しているかは、小さな入力で動かして確認する。

テキスト入力で不要なvision encoderをpackから除外できるか、公式のテキスト出力との一致で確認する。base全体のファイルサイズと、canisterへ載せる言語部分・readoutのサイズを分け、LoRAを統合する場合と実行時に適用する場合の容量・精度・命令数を比較する。

全重みの実バイト数、最大tensor、量子化の補助情報、tokenizer、推論時の状態を見積もる。4Bの単純な重み概算はINT8で約4GB、INT4で約2GBだが、これは実行メモリでも最終packサイズでもない。GiBとGBを区別して報告する。

成果物: MODEL_LOCK.jsonの形式設計、演算一覧、容量見積もり、利用する実装とライセンスの一覧。ロックにはbase・adapter・readout・tokenizer・processor・校正ファイル・公式serverの識別情報を個別に保存する。

## 段階2: canisterへ移す前に精度を比較する

固定した公式Imajev実装で4Bの参照出力を取得する。公式のMLXまたはPyTorch経路からこのマシンで利用可能な方法を選び、vLLMやCandleへそのまま読み込めるとは仮定しない。1質問・テキストのみ・rotations=1・対応するcalibration.jsonありを基準にする。baseのBF16、adapterとreadoutの実精度を記録し、量子化結果と分ける。

BOOM DAOの617など既存の問題を含め、最低投票期間、最大ロック期間、stake、mint、変更なし、情報不足を扱う評価集合を作る。提案ごとの事実、旧値の取得時点、質問、選択肢、正解の根拠を保存する。既存結果と同じ質問の比較と、情報を増やした質問の比較は別の列で報告する。

主な評価は、項目別の判定一致、重大変更の見逃し、誤警告、情報不足の扱い、logits差、確率差である。正解が定義できる集合だけで精度や校正を評価し、Layaとの一致率を正解率として扱わない。量子化やpromptの調整に使った問題と、最終評価用の問題を分離する。少数の提案だけで一般的な精度向上を主張しない。

成果物: 固定入力、参照logits、評価手順、Imajev-4B・Layaの比較表。JevK5-4BとPlumb-4Bは必要に応じてホスト上の比較へ加えるが、同時にcanisterへ移植しない。Imajevに改善が見られない場合は、全面移植より先に原因を調査して採用判断を見直す。

## 段階3: Rust/Wasmで演算を再現する

まず小さな合成tensorで個々の演算を検証する。その後、対象モデルの1ブロック、数ブロック、全体の順に参照実装と比較する。比較する中間tensor、許容誤差、対象の重み精度を事前に記録する。

最初は正しさを優先し、量子化誤差と移植誤差を混ぜない。全モデルを高精度でcanisterへ載せる必要はなく、ネイティブ側や小さなfixtureで高精度の比較を行ってから量子化版へ移る。

判定は公式promptのprefillと専用decision readoutで完了させる。Imajevの選択肢コード・unknown・読み出すhidden state位置・候補マスク・校正・APIへの変換を公式コードどおりに再現する。JevK5の回答文字の次token logit方式へ置き換えない。公開資料のreadoutは255個の選択肢コードとunknownからなる256出力であるが、採用revisionのshapeを実ファイルで検証する。必要なreadout行だけ計算する最適化は全readoutの参照結果と一致してから採用する。

型安全な出力も検証対象にする。許可した選択肢だけを返すこと、確率が有限かつ有効な範囲に収まること、確率の正規化とunknownの扱いが公式実装と一致すること、Scoreの値が定義したrubricに対応することを確認する。型の正しさを判定の正しさとして扱わない。

成果物: ネイティブ実行器、演算テスト、参照比較、Wasmでの小さな演算ベンチ。

## 段階4: 重みの配置と分割単位を決める

次の構成を順に実測して選ぶ。最初から複数canisterへ固定しない。

1. 4Bの量子化した言語部分の重みと専用readoutをheapへ保持する構成。LoRAの統合後の容量と作業領域まで含めて収まるか確認する。
2. 一つのcanisterのstable memoryへ全重みを置き、必要な層またはtensorタイルだけquery内でheapへ読む構成。
3. 層を複数canisterへ配置し、クライアントがactivationを次のcanisterへ渡す構成。

stable memoryに重みが収まっても、query内の読み出し上限、コピー、量子化の展開、演算の命令数が制約になる。対象subnetの現行上限と実測を使い、常に全重みをheapへ複製する実装を避ける。

分割は層境界を第一候補とする。1層だけでquery上限へ達する場合は、行列積のタイルやトークンブロックなど、層の内部を区切る必要がある。recurrent系ブロックの分割では内部状態と位置の継続も設計する。クライアントへ保持する情報を不要に増やさないため、まず層を順に進める方式を調べ、同じpromptに対する不要なdecode用cacheを作らない。

分割queryの状態には、モデル・revision・形式・位置・shape・入力の識別情報を持たせる。サイズ上限、展開上限、checksum、有限値、scale、進捗を検証する。別モデルや別入力の状態混在を拒否する。ただし、checksumや入力hashだけでクライアントによる改変を防げるとは扱わない。初期段階はowner専用の補助判定に限定する。

成果物: 構成ごとのheap・stable使用量、最大読み出し量、query命令数、通信量と、採用構成の理由。

## 段階5: 通信と計算を最適化する

重みの精度とactivationの通信形式を別々に比較する。重みはINT8を基準候補にし、容量が必要な場合はINT4を評価する。GGUFなど公開の量子化結果はホスト上の候補比較に使えるが、既存canisterの演算へそのまま読み込めるとは仮定しない。

中間状態はF32を比較基準にし、可逆圧縮、INT8、必要に応じた他の形式を比較する。重み量子化とactivation量子化を同時に導入せず、どちらが判定差を生んだか追跡する。

行列積での積和からscale適用、bias、丸めまでを可能な範囲でまとめ、不要なINT32・F32配列の確保を削減する。固定scaleや整数の乗算・シフトは校正データと未使用の評価データで検証してから採用する。Wasm SIMDの効果は命令数と時間の両方で測る。

通信が切れた場合に同じ状態から再開できるクライアントを作る。状態はバイナリで保存し、モデル識別情報とchecksumを付け、保存途中のファイルを有効なcheckpointとして扱わない。一時的な通信エラーは同じ要求で上限付き再試行し、命令上限では分割幅を縮小する。推論を自動的にupdateへ切り替えない。

成果物: 精度差・通信量・命令数・時間の比較と、採用した設定。

## 段階6: ローカルcanisterで通し測定する

IC-Imajev専用のローカルidentity、canister、ネットワーク設定を使う。既存のIC-Layaのローカル環境を流用して上書きしない。入力長はまず32・64・128 tokens、2〜7選択肢で確認し、成功後に256・512以上へ広げる。token数は自然文の文字数ではなく、公式promptと特殊tokenを含む実入力で数える。

各測定で、ソースcommit、Wasm hash、モデルrevision、pack hash、tokenizer hash、量子化設定、入力hash、分割幅、cache条件を保存する。

| 測定項目 | 報告する内容 |
|---|---|
| 判定 | 参照との差、選択肢順位、項目別の正解・誤り |
| 命令数 | 各query、最大値、全体合計、失敗分。handler内だけか全体かも明記 |
| 時間 | 各queryと全体、warm/coldの条件、複数回の分布 |
| 通信 | バイナリ要求・返信、成功分と失敗分、外側の通信を含むか |
| メモリ | heapとstable、準備中と推論中、観測ピークの測定間隔 |
| 初期準備 | upload量、update回数、warmup命令数、cycles差分 |

query上限の直前まで使い切る設定を標準にしない。初期の目安は各queryのhandlerを40億命令以下に抑え、Candidのdecode・返信encodeを含む実呼び出しの成功で確認する。分割後の最大命令数に加え、合計命令数や通信増加も評価する。

成果物: 再現可能なローカル推論、境界入力と再送・再開の検証、構成選択の報告。

## 後続の画像対応

テキスト経路が完走し、品質と費用を評価できてから画像対応を検討する。画像前処理、vision encoder、画像tokenへの変換、言語部分への接続、追加重みの配置を別の実装段階にする。画像tokenが増やす計算・中間状態・通信を測る。テキストだけの結果で画像経路の品質や費用を推定しない。

画像対応は今回のテキスト移植の完了条件に含めない。実施する場合は画像と構造化情報の矛盾、情報不足、二画像の比較など、実際の用途に対応した評価集合を作る。

## mainnetへ進む条件

ローカルで対象入力が完走し、参照との誤差と量子化による判定差が把握できてからmainnetの試験を検討する。実装開始時点ではmainnetデプロイを行わない。

mainnetでは、モデルを保持する費用、初回upload・warmup費用、必要なcycles残高と凍結閾値を見積もる。通常queryの現在の無料扱いだけを理由に費用や負荷を無視しない。ネットワークの待ち時間、同時実行、再試行の挙動はmainnetで別途測る。

最初の公開範囲はowner専用とする。upgrade後はstable memoryのモデルから実行用の状態を作り直せるようにする。補助判定をそのまま資金移動やDAO操作の承認へ使わず、その用途では認証・結果の検証・既存のupdate経路との関係を別途設計する。

## 最初に着手する作業

1. Imajev-4Bの評価済みrevisionと公式serverを固定し、base・adapter・readout・校正の組合せと容量表を作る。
2. このマシンで動く参照実装を選び、短い入力の選択肢logitsを取得する。
3. BOOM DAOの固定した比較問題で、Layaからの改善を確認する。
4. 最も移植が難しいブロックを小さなtensorでWasm実行し、命令数を測る。
5. その結果で4Bの重み常駐・stableからの部分読み込み・複数canisterのどれへ進むか決める。2Bへ縮小する場合は別の採用判断として記録する。

完了の判断は、モデルが大きいことではなく、判定品質・待ち時間・通信量・運用費用を実測して選べる状態になったことで行う。

## 2026-10-01 実装到達点

固定モデルの公式ホスト比較、Rust/Wasm移植、全層INT8 pack、client-held通常query分割とcheckpoint再開を実装した。BOOM DAO 617・132 token・rotations=1がローカルcanisterで完走し、参照likelyと一致。3,908 query・604秒・通信5.38 GB。詳細は `docs/FULL_INFERENCE.md`、各queryの実測は `docs/full-results.json`。全32層BF16 A/B、複数問題のINT8精度、通信圧縮は後続評価。画像/mainnet/push/PRは実施していない。

可逆BF16通信と大型queryにより、同じ質問を3,108 query・560.522秒・通信2.549 GBへ改善（通信52.64%減）。全層・最終hidden・判断logit/確率が改善前とビット一致。`docs/COMMUNICATION.md` と `docs/efficient-results.json` に実測を記録。
