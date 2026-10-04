2026-10-04: 実装レビューを修正し、層境界の連結経路を全体へ接続。主87/情報不足80は62→54 query・全出力ビット一致。89 tokenは入口超過を保存し標準62queryへ切替。主の総命令3.11%・通信9.12%増のため実験オプションで、50/32未達。全6条件回帰・最終3条件と54query再開を確認。[実測と修正](ROLL_QUERY.md)。

2026-10-04: 部分down前倒しを追加。87 token/1600行と89 token/1280行でcarry/finish一致、89 token/1600行は5B超過。全6条件回帰も通過。全体graphは未接続で主62 query、50未達。[実測と次のボトルネック](MLP_DELTA_STREAM.md)。

2026-10-04: MLP完了→次層Delta部分→次queryのDelta完了/次層MLP準備を実装。39成立条件でcarry/次層finishまでビット一致、45条件は5B超過を記録。新Wasmの全6条件回帰も通過。全体は未接続で主62 query、50未達。詳細は [MLP_DELTA_STREAM.md](MLP_DELTA_STREAM.md)。

2026-10-04: レビュー修正を `597206a` にコミット。続いてMLP最終chunkとfinishのquery融合を実装し、45実測条件でhidden/normビット一致、分割比1 query・約92〜105M命令・約2.86〜3.18MB通信を削減。全体graphへは未接続で、主62 query。詳細は [MLP_COMPLETE.md](MLP_COMPLETE.md)。

# Imajev-4Bの実装状況

2026-10-04レビュー：compact tailの古いlayer30出力を除去し、MLP carry送信時の検査用全復元をRust/Pythonとも省略、frame versionの型検査を修正。候補/通常featureのRust・Python検証、保存payload315個のbyte一致、Wasmビルドを確認。実canister命令の再測定は後続に分離。主62 query・50未達を維持。 [レビュー記録](IMPLEMENTATION_REVIEW_2026_10_04.md)。

2026-10-04追加：列継続のS1係数変換を新しい列だけへ限定し、余分なゼロ埋めも除去。実32条件でbit一致・命令減・通信不変、主87の2分割225万/4分割675万命令減。Delta16+6+10 headの実12条件でgated/conv履歴が一致、profileで入力量子化/Aは最初だけと確認。全6条件のgraph回帰も一致。ただし新APIは現graphへ未接続で主62 query、50/32未達。 [実装と実測](INT8_COLUMN_CONTINUATION.md)。

2026-10-03追加：Delta out projectionを列単位で継続し、新しい列だけをINT8量子化・base/F32 A加算、finishではF32 Bだけを実行。実8入力×4分割の32条件で一括出力とbit一致、全6条件のgraph回帰も一致。現graphは未接続で主62 queryのまま、分岐追加で1,241,192命令増。単独分割はquery/通信増のため性能改善として未採用、50/32未達。 [実装と実測](INT8_COLUMN_CONTINUATION.md)。

2026-10-03追加：入力量子化・gate/up F32 Aを一度だけ計算し、product量子化とdown F32 Aの部分和を保持するMLP分割APIを実装。80/87/89 token・実3層・5分割の45条件で準備状態/hidden/normが既存INT8版とbit一致、profileで再実行なしを確認。既存graphは未接続で主62 queryのまま、API分岐追加により8,528命令増。単独分割は通信/命令増のため性能改善として未採用、次はDelta分割との融合。50/32未達。 [実装と実測](MLP_STREAM.md)。

2026-10-03追加：分割MLPに残っていた固定F32重みの元配置復元と入力転置を除去。cold132の全推論で1,447,149,192命令（0.400601%）減、全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致。主87は62 query・命令・通信とも同じで50/32未達。F32列部分和をクライアント保持し加算順序を保って継続する部品も、実3層・251診断queryで一致を検証。 [分割MLPの実測](MLP_PREPARED_CAPTURE.md)、[F32継続演算](F32_COLUMN_CONTINUATION.md)。

2026-10-03追加：整数S1の出力tileを32→128行へ広げ、同じ入力ロードとtoken/scale/address処理を共有。主さらに7,888,349,165命令（3.245%）減、235,201,564,012命令。全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致、失敗/replay0。最大変更89は63→62query、主は62で50/32未達。主単回時間20.221→35.584秒の悪化を記録。256行版はlocals上限でinstall拒否、64/128診断は全11入力一致。 [実装と実測](S1_OUTPUT_TILE_REUSE.md)。

2026-10-03追加：各queryの入力・出力frameハッシュを、署名検証するクライアントへ移管。主さらに2,021,749,972命令（0.825%）減、243,089,913,177命令。全6条件の返却hidden/保持state/判断/確率が既存INT8版とbit一致、失敗/replay0。62 query・通信は同じで50/32未達。署名改ざん拒否・保存状態破損の送信前拒否・不正payload48件拒否を検証。89-token tail統合は5B超過で分割維持。 [実装と実測](HOST_FRAME_CHECKSUM.md)。

2026-10-03追加：block codecの有限値検査とBF16分類を一走査にし、復号のゼロ埋め・再走査を除去。主さらに788,566,976命令（0.321%）減、245,111,663,149命令。全条件の返却hidden/保持state/判断/確率bit一致、失敗/replay0。62 query・通信・観測heap終了最大は同じで50/32未達。100 codec診断と正しいchecksum付き不正payload32件の拒否も検証。[実装と実測](CODEC_STREAMING_CHECK.md)。

2026-10-03追加：量子化で入力の有限値検査とblock256の最大値探索を一走査へ統合。主さらに1,233,003,258命令（0.499%）減、245,900,230,125命令。全条件の返却hidden/保持state/判断/確率bit一致、失敗/replay0。62 query・通信・観測heap終了最大は同じで50/32未達。単体36通常queryで整数/scale一致とInf/NaN拒否も確認。[実装と実測](QUANTIZE_PEAK_SCAN.md)。

2026-10-03追加：最終2層で検査済み入力を型で引き継ぎ、接続時の全走査・連結コピーと不要なlayer30 hiddenの返送を除去。主63→62 query、51,181,409命令・Candid1,342,937 bytes減、247,133,233,383命令。全条件の返却hidden/保持state/判断/確率bit一致、graph失敗/replay0。layer30 hidden非返却を明示、87以下に限定、89/132は分割経路。50/32未達。[実装と実測](TERMINAL_TAIL.md)。

2026-10-03追加：S1の7係数で繰り返す共通byte offsetをtoken pairごとに一度計算し再利用。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主632,813,440命令（0.25535%）減、247,184,414,792命令。63 query・通信・観測heap終了最大は同じで50/32未達。固定pointer再利用も単体で出力一致だが1token増加により未接続。[実装と実測](S1_ADDRESS_REUSE.md)。

2026-10-03追加：LoRA Aのinput転置を省き、元容量の固定output32配置とK64部分和で全tokenのweight loadを共有。全5条件の保持hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに24.683億命令（0.9834%）減、248,528,396,096命令。64 query・通信は同じで50/32未達。主MLPを再profileし、base77.58%、A合計4.61%、B7.42%を実測。[F32_A_REUSE.md](F32_A_REUSE.md)。

2026-10-03追加：S1入力係数の全領域ゼロ埋め直後の全上書きを除去し、一度だけ書き込む。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。主353,828,218命令（0.14257%）減、247,817,228,232命令。63 query・通信・観測heap終了最大は同じで50/32未達。S2共通変換再利用3候補は現行S1より増加し未採用。[実装と実測](OPERAND_WRITE_ONCE.md)。

2026-10-03追加：最終Attention/MLP/normと専用F32判断を1通常queryへ統合し、最後のhidden再送を除去。主64→63 query、495,771命令・Candid10,619 bytes減。全5条件の保持hidden/state/判断bit一致、graph失敗/replay0。同じmoduleの2→1呼び出しcontrolとqueryなしcheckpoint再開も検証。固定payload・観測heap終了最大を維持。50/32未達。[実装と実測](TERMINAL_DECISION.md)。


2026-10-03追加：INT8固定scaleのqueryごとの読み出し・byte→F32復号・正値/有限値再走査を除去。固定payload不変、全5条件の保持hidden/state/判断bit一致、失敗/replay0。主356,843,875命令（0.1436%）減、248,171,552,221命令。64 queryと通信は同じで50/32未達。観測heap終了最大は+2,424,832 bytes。第二候補32a3e1d5…を実験用基準とする。[固定scaleの実装と実測](FIXED_SCALE_REUSE.md)。


2026-10-03追加：固定F32 LoRA Bを元容量のoutput32配置にし、全tokenでweight loadを共有、query input転置を除去。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主さらに46.333億命令（1.8125%）減、250,996,660,791命令。cache4.065 GBを維持、64 query・通信は同じで50/32未達。LoRA AのK64診断も主18.738%減・22測定queryでdigest一致し、次の全モデル接続候補とする。[F32_WEIGHT_REUSE.md](F32_WEIGHT_REUSE.md)。

2026-10-03追加：元容量のINT8四象限配置を一度準備し、query内のS1入力変換・初回weight展開を共有。全5条件の全層hidden/state/判断が既存INT8版とbit一致、失敗/replay0。主50.717億命令（1.9454%）減、全条件0.804〜2.569%減。固定cache容量4.065 GBを維持。主64 query / Candid124,634,637 bytesで50/32未達。詳細と再現：[STRASSEN_RAW_REUSE.md](STRASSEN_RAW_REUSE.md)。

2026-10-03追加：整数S1の水平和を出力再構成後へ移し、input/weightを最初のdotでlocal.teeして読み直しを省いた。主87-token Qは29,617,156命令（2.8360%）減、全5実入力＋6境界でnative digest一致。ただしprefix45は0.4737%増、固定係数3.5倍・全モデル4GiB配置は未解決で未採用。50/32未達。[WAT_S1_REDUCTION_REUSE.md](WAT_S1_REDUCTION_REUSE.md)。

2026-10-03追加：carry復号を2 symbol/4 byte単位にし、弱圧縮planeをrawへ変更。実87-tokenでdecoder単体約60%削減・全byte一致。512行分割のMLP→Delta融合は3箇所で4.922〜4.923B命令・bit一致で初完走。ただし2 queryの総命令約2.3%増、通信約116〜117万bytes増で既存graphには未採用。全5条件の回帰はbit一致、主64 query / 260,701,702,281命令。50/32未達。[CARRY_PAIR_DECODER.md](CARRY_PAIR_DECODER.md)。

2026-10-03追加：DEFLATEの代わりに可逆raw/constant/Huffman carryを実装。87-token frame約173〜175万bytes、6 groupのbyte一致、partial MLP出力bit一致。45-token診断で融合約2953〜3488万命令減、復号約12.8〜14.3%減。87-token融合は5B超過で未採用、最新完走graphは64 queryのまま。詳細：[CARRY_HUFFMAN.md](CARRY_HUFFMAN.md)。

2026-10-03追加：MLP down先頭256行を前queryへ移し、残りと次Deltaを可逆carryで融合する実装を追加。87-tokenは3箇所すべて5B上限超過で未採用。partial finishは通常MLPとbit一致、45-token融合もbit一致。profileで復号のうちinflate1.58〜1.62億命令を特定。単純Huffman/raw/constantのframe見積り1.75〜1.77 MBを得て次のdecoder候補とする。既存全5条件はbit一致・失敗/replay0、主64 query。50/32未達。[実装・失敗・内訳](MLP_DELTA_FUSION.md)。

2026-10-03追加：Delta初期状態をkey優先のまま渡し、復元後の転置・その逆転置とhead作業buffer確保を省く。全5条件のhidden/stateと判断/確率が既存INT8版とbit一致。主260,701,622,342命令、前回から767,186,917命令（−0.29341%）減。64 query・Candid124,634,637 bytesは同じ。候補module1f445089…、主/Layaは変更なし。MLP完了＋次hybrid Deltaの個別query合計は3箇所で5B内、融合の実測とquery数削減は未実装。50/32未達。[実装と測定](DELTA_STATE_LAYOUT.md)。

2026-10-03追加：使わないDelta最終状態の転置・書き戻しを省略。全5条件の保持hidden/stateと全判断出力が既存INT8版とbit一致、失敗/replay0。主問題は261,468,809,259命令、直前候補より695,016,960命令（−0.26511%）減。64 query・通信124,634,637 Candid bytesは同じ。通常の分割Deltaとprefix記録は状態書き戻しを保持。小型Wasm24通常queryでも出力一致・scratch未更新を検証。全model専用module3dbdf889…、主canister/Layaは変更なし。50/32未達。[実装・測定](DELTA_DISCARDED_STATE.md)。

2026-10-03最新：固定INT8配置を準備updateで置換し、query内の入力変換を共有。分割queryの8行境界でも元byteへ読み戻さず、全5条件で保持hidden/state/判断/確率が一致、失敗/replay0。主65.21億handler命令（2.3933%）減、全条件2.39〜7.78%減。主67/初回133 query・通信量は同じ、50/32未達。cache容量は同じ4.065 GB、一度の準備は609.94億命令へ増加。単回時間は主・prefixなし等で悪化し速度改善は未確認。採用module36c04a57…、build時`experimental-prepared-output-pairs`、準備・検証時`--require-output-pairs`を追加。 [実装・全5条件実測](OUTPUT_PAIRS.md)。

2026-10-03追加：毎質問のprefixログ復元を減らす可逆hybridを専用canisterへ実装。全24層・48通常queryで保存状態/native/Wasm bit一致、復元部分28.4〜28.6%減（計11.91億命令）。通信は増えるため全推論へ未採用。主67 query、50/32未達。[実装と範囲](PREFIX_HYBRID.md)。固定重みのI16安全窓候補は22通常queryでbit一致したが主Q命令38.94%増のため不採用。[結果](BOUNDED_I16.md)。


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


2026-10-02最新：DeltaのQKV/Z projection・gate・conv/再帰を通常queryへ統合し、量子化・元F32の2つのA積・gateをclient-held capture/reuseで一度だけ計算。全5実行で従来Wasmの保持hidden/state/判断/確率bit一致、失敗/replay0。主問題265→193 query、通信459.92→371.33 MB（19.262%減）、命令3234.18→3219.11億（0.466%減）。初回prefix込み507→387、prefixなし438→294 queryで50/32未達。採用modulecfa16f56…、`--fuse-delta-projected`を明示する。既存native/Wasm gate差は分離して確認し、native同士とWasm同士で一致を検証。整数dotの演算方式が次の対象。 [実装・全層比較](DELTA_PROJECTED.md)。

2026-10-02最新：AttentionのK/Vで量子化を共有し、K norm/RoPEとQ projection→norm/RoPE→GQA→gateを通常queryへ統合。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題305→265 query（13.115%減）、通信506.37→459.92 MB（9.173%減）、命令3247.65→3234.18億（0.415%減）。初回prefix込み587→507、prefixなし485→438 queryで50/32未達。単回時間には増加例があり速度改善は主張しない。採用module58cf6426…、`--fuse-attention`を明示する。次のDelta投影統合は5条件の実packetが2MBに収まることだけを確認し、kernelは未実装。 [実装・全層比較](ATTENTION_FUSION.md)。


2026-10-02最新：毎queryで二重に行っていたclient request encode・hashを1回にまとめ、状態frame checksumをversion2/BLAKE3 SIMDへ軽量化。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。主問題3547.05億→3247.65億命令（8.4408%減）、prefixなし5194.94億→4709.36億（9.3472%減）。query数305／初回587／prefixなし485、通信量は同じで50/32未達。固定重み準備はmodel変更/upgrade時のみ、推論は通常query・中間状態はclient-heldを維持。採用module412c565e…、旧frame互換を残し`--frame-checksum blake3`で明示する。 [実装・全層実測](FRAME_CHECKSUM.md)。
2026-10-02最新：分割MLPでもblock256量子化とgate/up両方の元F32 LoRA Aを再利用し、31組の再計算を省いた。全5実行で保持hidden/state/判断/確率bit一致、失敗/replay0。prefixなし132-tokenは追加31.57億命令（0.6040%）減の5194.94億、485 queryは同じ、通信4.51 MB増。主問題は再利用対象0で305 query・3547億命令、初回prefix込み587 query。50/32未達。採用module752a35b4…、再利用は明示オプション、重み準備はmodel変更/upgrade時だけ。 [実装・全層比較](MLP_REUSE.md)。

2026-10-02追加探索：固定weight scaleの復元・配列確保・正の有限値の再走査を準備updateへ移す候補を実装し、全5実行で保持hidden/state/判断/確率bit一致を確認。ただし主問題は命令0.0385%増、全5実行で増えたため不採用。検証済みソースarchiveと差分を保存し、運用ソース31 hash・主canisterを前の直接INT8展開版e961dda8…へ復元。query数305、初回587、prefixなし485、50/32未達は同じ。F32状態の可逆圧縮も3形式を実測したが、最良でも平均1.83 MB/stateで87-token入力と2 MBへ収まらず、本番codecには未接続。 [候補と全層比較](PREPARED_SCALES.md)。


2026-10-02追加探索：分割間の入力量子化とLoRA Aをclient-held stateで再利用するcapture/reuseを実装。2queryを維持し、2候補・168成功queryで独立scalarとbit一致、不正状態4件を拒否。ただし132-token投影はSIMD版でも命令1.285%増・通信21.44%増で採用保留。整数のまま渡す可逆codecを試作し、次はWasmで検証する。全層305 query・初回587・prefixなし485、50/32未達は変わらない。 [実装・比較](PROJECTION_REUSE.md)。

2026-10-02追加探索：Strassen係数の展開・全要素検証を準備updateへ移し、毎SIMDロードの行アドレス計算も共有した。3候補・合計162通常queryで独立scalarとbit一致したが、87-token base Qは候補16.40億・通常INT8 14.36億命令で不採用。主canisterは既存検証済み版と721 tensor cacheを保持。全層3547億命令・305 query（初回prefix込み587、prefixなし485）と50/32未達は変わらない。 [探索結果](STRASSEN_PREPARED.md)。

2026-10-02最新：元F32 adapter等も準備updateで復元・有限性検証し、通常queryでは同じ配列を共有。主問題3584.2億→3547.0億命令（さらに1.038%減）、全5条件でhidden/state/判断/確率bit一致。stepのstable readは488 MB→0.566 MB、305 query・通信506 MBは同じ。初回prefix込み587 query、prefixなし485 query。重み準備は別途721 update・150.28億命令・234.703秒、cache4.065GB・終端heap最大4.115GB。50/32 query未達。 [PREPARED_F32.md](PREPARED_F32.md)。

2026-10-02前段階：固定INT8 dense重み272個（3.575GB）を準備updateで一度保持し、通常queryは借用して読み出し・バッファ確保・コピーを省く。主問題3654.5億→3584.2億命令（1.923%減）、全5条件でhidden/state/判断/確率bit一致。305 query・通信506 MBは同じ、初回prefix込み587 query。準備は別途272 update、94.36億命令・85.521秒。heap最大観測3.600GB、selected local heap limitは4GiB。50/32 query未達。 [BORROWED_WEIGHTS.md](BORROWED_WEIGHTS.md)。

2026-10-02前段階：量子化済み入力を変更不能な型にし、投影ごとの全走査による再検証を除去。主問題3675億→3655億命令（0.560%減）、305 query・通信506 MBは同じ。全5条件で保持hidden/state/判断/確率bit一致、失敗/replay 0。prefixなし132 tokenは5370億命令（0.742%減）・485 query、初回prefix込み587 query。50/32 query未達。 採用Wasm `e1be06bc…`、Rust40＋compile-fail doctest通過。[QUANTIZED_INPUT_REUSE.md](QUANTIZED_INPUT_REUSE.md)。

2026-10-02前段階：RoPEの周波数・sin/cosと固定A_logのexpを共有し、Q/K正規化をRoPE queryへ統合。主問題321→305 query、通信519→506 MB、3691億→3675億命令。全5実行で保持hidden/state/判断bit一致、失敗/replay 0。prefix準備298→282 query、初回合計587 query。cacheなし132 tokenは501→485 query。50/32 query未達。 採用Wasm `fe276255…`、Rust39/Python42 tests通過。[ROPE_REUSE.md](ROPE_REUSE.md)。固定adapter準備時統合のホスト候補は23/23判断一致だが、丸めと量子化が変わり確率差最大7.64ポイント。canisterの採用版には含めない。

2026-10-02前段階: 同じQ/Kの正規化・同じ入力のactivation量子化を再利用。主問題369,120,738,555命令（さらに0.589%減）、321 query、通信519 MB。全5実行で保持hidden/state/判断bit一致、失敗/replay 0、前後認証module hash一致。Wasm `3a76af56…`、Rust36/Python42 tests通過。初回prefix込み619 query、cacheなし501 queryで50/32 query未達。[DELTA_REUSE.md](DELTA_REUSE.md)。Strassenの補正を一度復元する候補は通常積和に負け不採用。[STRASSEN_WIDE.md](STRASSEN_WIDE.md)。

2026-10-02前段階: 可逆block256通信を実装。主問題321 queryのまま、通信550→519 MB（5.576%減）、命令3773億→3713億（1.589%減）。全5実行で保持hidden/state/判断bit一致、失敗/replay 0、前後認証module hash一致。Wasm `02c4243b…`、Rust36/Python42 tests通過。実入力25形状の計測後、通常版への復帰を認証済みmodule hashで確認。整数dot/LoRA積和が主要対象、50/32 query未達。[BLOCK_CODEC.md](BLOCK_CODEC.md)。

2026-10-02前段階: 残差/normをMLPへ統合。主問題352→321 query、通信579→550 MB、3773億命令。prefix準備330→298 query、初回合計619 query。cacheなし132 tokenは従来経路501 query。全5実行で保持hidden/state/判断bit一致、失敗/replay 0、前後認証module hash一致。既定Wasm `821f969e…`、Rust34/Python40 tests通過。50/32 query未達。[MLP_NORM_FUSION.md](MLP_NORM_FUSION.md)。次の統合に向けた可逆block codecはpacket試作のみで未実装。

2026-10-02前段階: 整数結果のSIMD取り出しとDelta係数共有を拡大。主問題3891億→3795億命令（さらに2.48%減）、352 query・通信579 MB。全5実行と整数126条件でbit一致、失敗/replay 0、前後認証module hash一致。既定Wasm `7c366d0b…`、Rust33/Python40 tests通過。単回の実時間は増えた例もあり速度改善とは報告しない。50/32 queryは未達。[PACKED_REDUCTION.md](PACKED_REDUCTION.md)。

2026-10-02前段階: Delta再帰の係数共有を採用。主問題4053億→3891億命令（さらに3.99%減）、352 query・通信579 MB。新prefixを含む全5実行で保持hidden/state/判断bit一致、失敗/replay 0、前後認証module hash一致。既定Wasm `ec3ebc93…`、Rust33/Python40 tests通過。50/32 queryは未達。[DELTA_SIMD.md](DELTA_SIMD.md)。事前変換Strassenは87-token Q投影の命令が約2倍となり不採用。[STRASSEN_PREPACKED.md](STRASSEN_PREPACKED.md)。

2026-10-02前段階: experimental-token-scaleを無効にしたWasm `8a84a27e…` で新prefixを含む全5実行を再検証。主問題352 query・4053億命令（10.06%減）、最大変更383→352 query・4185億（11.30%減）。保持hidden/state/判断bit一致、失敗/replay 0、前後認証module hash一致。Rust30/Python40 tests通過。50/32 queryは未達。[KERNEL_UNROLL.md](KERNEL_UNROLL.md)。

2026-10-02最新: 整数タイルの結果取り出しをさらに改善し、短い入力の余分なpadding演算を除去。主問題352 query・4053億命令（terminal-v3比10.06%減）、最大変更352 query・4185億（11.29%減）。126条件native/Wasm一致、全層5実行bit一致・失敗/replay 0。別のtoken-scale候補はホスト23/23ラベル一致、確率差最大10.9ポイント。既定の演算はblock256。50/32 queryは未達。[KERNEL_UNROLL.md](KERNEL_UNROLL.md)。

2026-10-02追加: 改良カーネルでは89-token MLPの全列が5B内に収まり、最大変更を383→352 queryへ短縮。主/情報不足/最大変更は全て352 query。最大変更の命令は4718億→4238億（10.17%減）、通信606.35→591.31 MB。新prefixを含む全5実行でbit一致・失敗/replay 0・前後module hash確認。目標回数は未達。詳細 [KERNEL_UNROLL.md](KERNEL_UNROLL.md)。

2026-10-02追記: 整数dotの結果取り出しとDelta状態store削減を全層検証。主問題は352 queryのまま、4507億→4098億命令（9.07%減）、保持hidden/stateと判断bit一致。3問題・prefix・cacheなし全5実行は失敗/replay 0。最大lock問題の既存誤判定は残る。詳細は [KERNEL_UNROLL.md](KERNEL_UNROLL.md)。50/32 queryは未達。

# 2026-10-02 実装・測定状態

**最新の追加改善（2026-10-02）：最終tokenへの演算省略・終端MLP統合・QKV/ゲート射影統合を実装。45-token prefix準備済みの132-token入力は379→352 query、命令4655億→4507億、通信615→579 MB。判断、前31層の全hidden、最終層の最後のhidden、保持stateが従来INT8版とbit一致。prefix準備は別途330 query。32 queryは未達。** [実測と省略範囲](TERMINAL_READOUT.md)。

**前段階の追加改善（2026-10-02）：Delta演算・ゲート射影の統合と短いMLPの分割削減により、共通45-token prefix準備済みの132-token入力は603→379 query、命令4854億→4655億、通信807→615 MB。全32層のhidden・保持state・判断が従来INT8版とbit一致。prefix準備は別途354 query。32 queryは未達。** [実装・検証・残る制約](FUSED_STAGES.md)。

**前段階の追加改善（2026-10-02）：終端Delta状態の返送省略とGQAのK/V共有を実装。通常132-token入力は708→700 query、命令1.05%減・通信5.27%減。共通prefix準備済みでは611→603 query、命令1.46%減・通信7.53%減（807 MB）。準備は別途570 query。3入力で全32層hidden・保持した状態・判断が旧INT8版とbit一致。50 queryは未達。** [実装・測定・再現](COMPACT_HEADS.md)。

追加探索：最新dotの内部計測、回転W8A8の3入力・414標本試験、終端Delta状態とGQA通信の省略余地を記録した。[EXPLORATION.md](EXPLORATION.md)。採用版の性能値は以下のままで、探索結果を新規の全層高速化として計上しない。

**前段階の追加改善（2026-10-02）：INT8直接ロードとtile変更で、cacheなし708 queryの命令数を7393億→7110億へ3.82%削減。共通45-tokenのclient cache利用時は611 query・4926億命令・0.873 GB。ただし初回cache準備は別に578 query・2557億命令・0.464 GB。3入力で全32層hidden/state/判断が以前のINT8版とbit一致。50 queryは未達、速度向上は未確認。** [複数方向の実装・不採用候補・初回費用](DIRECTIONS.md)。

前段階：演算方式変更の許可を得て、整数base＋元F32 LoRAを全32層へ接続した。617・132 tokensは1.022兆命令・932 query・1.671 GB・単回107.274秒。最初の全層から命令数72.24%減。ホスト23件で同じpackのF32/整数ラベル一致、元順序の正解付き7問は双方6/7。ただし確率差最大約8.48ポイントで、精度・校正の一般保証はしない。50 queryは未達。[INTEGER_ARITHMETIC.md](INTEGER_ARITHMETIC.md)。

数値を変えない追加改善も全32層で検証し、1,588→1,116 query、2.264→1.762兆命令。全hidden/state/logit/確率bit一致。[追加改善](FIFTY_QUERY_ANALYSIS.md)。以下は各段階の履歴。


INT8 base＋F32 adapter/readoutで、embeddingから全32層・専用readoutまでローカルcanisterの通常queryで完走した。132 token・BOOM DAO 617・rotations=1は3,908 query、604.274秒、合計3.682兆handler命令、Candid通信5.382 GB。参照と同じlikely、候補logit最大差0.04657、校正済み確率最大差0.001207。中間状態はclient保存、量子化追加なし。公式hiddenを計算入力へ流用していない。

再現と制約は [FULL_INFERENCE.md](FULL_INFERENCE.md)、各query実測は [full-results.json](full-results.json)。以下の部分演算測定は開発時の原型記録として残す。全層のINT8差には未量子化Rust/MLXの積和差も含まれるため、完全32層BF16 A/Bは残作業。

追加の通信改善で3,108 query・2.549 GB・560.522秒へ減らした（通信52.64%減）。全32層・final hidden・判断logit・確率が改善前とビット一致。最大queryは30.59億handler命令、終端heap最大観測87.1 MB。詳細は [COMMUNICATION.md](COMMUNICATION.md)、各queryは [efficient-results.json](efficient-results.json)。

追加でactivation INT8通信を実装し、3,108 query・1.285 GB・514.639秒で完走。likelyは一致するがBF16通信版との候補確率差は最大0.120719。精度同等とは言えない。head融合と32 query目標の制約は [ACTIVATION_INT8.md](ACTIVATION_INT8.md)。

head融合版を全32層で実測：2,132 query、461.531秒、Candid通信1,283,895,563 bytes、最大query命令数3,022,101,308、終端heap最大観測86,441,984 bytes。融合前INT8版と全層・state・判断のbit一致を確認。32 query目標は未達。現在の演算量では通常query上限内に1層を丸ごと置けない。[fused-results.json](fused-results.json)。

Layaの最新レポ/整数タイルを参照し、F32射影の16行ロード共有とpointer整理を追加した。全32層の総命令数3.133兆→2.329兆（25.66%減）、2,132 query、1.284 GB、単回251.002秒。全hidden/state/logit/確率がhead融合版とbit一致。49形状のraw F32 bit比較も成功。構造・実装・LoRA再計算の分解は [LAYA_COST_ANALYSIS.md](LAYA_COST_ANALYSIS.md)。32 queryは未達、演算の整数化を全射影へ接続する作業は残る。

最新の精度維持改善は [EXACT_OPTIMIZATION.md](EXACT_OPTIMIZATION.md)。元タイルの量子化境界を保つ2タイル融合とAttention再計算削減を実装。可逆通信の全層・state・判断が以前とbit一致で命令数32.0%減/1,588 query/1.984 GB。INT8通信もbit一致で1,540 query/1.014 GB。32 queryは未達。

## モデル固定

- adapter: `mohit67890/imajev-4b@c9e5f132465da85d31735ec502d5557982671a7d`
- base: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- 公式server: `mohit67890/imajev@a0134749e0900189c129cd6bb5000969f3b64bb5`
- baseのsnapshotをadapter_configの学習パスから特定。`revision:null` をmain参照として扱っていない。
- LoRA rank64、alpha128、scale2、言語層のq/k/v/o、gate/up/down、in_proj_qkv/z、out_projが対象。未統合参照を使用。
- 実readoutはF32 `[256,2560]`。manifestのcode/token bindingを公式backendが検証する。unknownは候補数に応じた最後の行であり、常にreadoutの最終255行を使うのではない。
- tokenizer、processor、専用readout、adapter、校正を個別にSHA256固定。`calibration.json` は温度1.3051569717552742、unknown offsetなし。calibration-rot4.jsonを使っていない。
- 本revisionのprompt layoutはstandard。最終`</think>\n\n`のhiddenから専用readoutを計算。次tokenのLM語彙logitへ置換していない。

全ファイルSHA256と実tensor配置は [MODEL_LOCK.json](../MODEL_LOCK.json)、容量集計は [capacity.json](capacity.json)。公式のモデルカードmainからshapeを推測していない。

| 実ファイル/領域 | bytes | GiB |
| --- | ---: | ---: |
| base全shard（header含む） | 9,319,828,096 | 8.680 |
| 言語tensor（BF16主体、一部F32） | 8,411,510,272 | 7.834 |
| vision tensor | 667,028,480 | 0.621 |
| MTPなどその他tensor | 241,199,104 | 0.225 |
| PEFT adapter | 487,648,432 | 0.454 |
| 専用readout | 2,621,520 | 0.00244 |
| embedding単体 | 1,271,398,400 | 1.184 |

言語部分は4,205,751,296 parameters。INT8の単純なparameter bytesだけで3.917 GiBとなる（実pack容量ではない）。scale・未統合LoRA・activation・decoder・allocatorを含めて4 GiB heapへ常駐できるとは扱わない。全層のtext packからvision/MTP/tied LM headを除外した。ホストtext-only全層captureの最終hiddenは既存参照とビット一致。

32層＝24 Gated DeltaNet＋8 full attention、hidden2560、MLP9216、語彙248320。必要演算はGQA/gating、64次元partial RoPE、QK RMSNorm、causal depthwise conv(kernel4)、QK正規化、F32 recurrent state、exp/softplus/sigmoid、gated RMSNorm、SwiGLU、residual、専用readout、候補mask、校正softmax。通常のattentionだけの移植では足りない。Candle/vLLMはこの実験で動作検証していない。

## 判断の比較

入力と正解根拠は [benchmarks/cases.json](../benchmarks/cases.json)。Layaは保存済み同一質問・選択肢・構造化要約を使用し、ソース、Git、identity、network、稼働canisterを変更していない。追加問題にはLayaの結果がないため、両モデルの正解率比較はできない。

以下は公式server shared-prefix経路、1質問・rotations=1。full-forward経路も別途取得。両経路の選択結果は23/23一致するが確率には小差がある。全結果と入力tokensは [host-results.json](host-results.json)。

| 問題 | Laya保存済み | Imajev（元順序） | 観察 |
| --- | --- | --- | --- |
| 617 最低投票delay 20,000倍 | unlikely | likely (0.8140) | 重大変更の見逃しを改善する候補 |
| 620 最低投票delay 2倍 | unlikely | likely (0.5161) | 他順序ではpossible。危険度goldは未定義 |
| 653 1口座に250M mint | likely | likely (0.4410) | unknownも0.2164。既存supplyなし |
| 最低投票delay増加の事実 | 未測定 | yes | 根拠一致 |
| 最大lock duration 1,000倍増加 | 未測定 | **no** | **誤判定・変更の見逃し**、2順序とも誤り |
| stake 1,000倍増加 | 未測定 | yes | 根拠一致 |
| 新規mintの事実 | 未測定 | yes | 根拠一致 |
| 変更なし | 未測定 | no | この事例では誤警告なし |
| 旧値なし | 未測定 | abstained | unknown=0.99815 |
| maximum増加、minimum不変 | 未測定 | no | 区別はできた |

根拠を明記した7問題は元順序で6/7。これらは今回用意した小さな診断集合で、一般精度や校正性能を表すベンチではない。変更量から投票参加・集中への将来の因果効果は一意に決まらないため、歴史3件にaccuracyのgoldを付けない。並べ替えは選択肢だけをcyclicに入れ替え、unknownは最後、各回独立rotations=1で平均なし。620で順序依存を確認した。

LayaとImajevのprompt/tokenizerが異なるため入力token数・raw logits・確率を直接同じ尺度として比較しない。Imajevの歴史3件は132/124/122 tokensである。

## 演算効率化の追加測定

Layaの実コードを参照したSIMD、base/LoRA融合、候補行readout、クライアント分割を実装した。132 token・256出力行のQKVで総命令数85.2%減、通信量56.9%減。旧新の各F32積和とBF16出力はビット一致。詳細・制約・追加候補は [OPTIMIZATIONS.md](OPTIMIZATIONS.md)、各query実測は [optimizations.json](optimizations.json)。以下の原型測定は以前のWasmの記録として残す。

## Rust/Wasmの実測

測定APIは通常query。中間状態はclient保存binary（model lock hash、pack hash、input hash、version、step、op、shape、checksum、有限値検査）。query間でheapのactivationを永続化しない。重みはstable、tensorの出力行タイルを必要分だけheapへ読む。INT8のrow scaleは別rangeから読む。

- 原型はreadoutの全256行を計算。改善版`decision_fast`は候補数＋unknownの行だけ計算する。
- readout重みINT8＋activation F32、重みF32＋activation INT8を独立に比較する。
- 以下のreadout/QKV原型のactivation INT8試験はF32へ復元して送った数値誤差試験。後続の全層ではINT8 binary通信を実装・測定済み。
- 256行の実QKV base射影、rank64のF32 LoRA A/B、BF16境界丸めをcanister queryで計算。
- 実第1層DeltaNetの1 value head（128×128 state）をnativeとWasmで計算。全32 heads/全24層の検証ではない。

[canister-check.json](canister-check.json) と [projection-check.json](projection-check.json) に各queryの命令数、要求・返信bytes、時間、heap/stable、pack/Wasm hash、元ログのhashを保存する。命令数はhandler内のstate検証、stable読み出し、展開、演算、内部状態encodeを含み、CDK Candid decode/encodeを含まない。通信は成功分のCandid request＋reply、HTTP/CBOR/signatureを含まない。query cache未制御、単回測定であり時間は計算速度の証拠にしない。heapはhandler終端のpage数、瞬間ピーク測定ではない。

| 検証 | 実測 |
| --- | --- |
| readout F32 / 23入力 | 選択一致23/23、logit最大差1.63e-5、確率最大差2.41e-6 |
| readout weight INT8 / activation F32 | 選択一致23/23、logit最大差0.01836 |
| readout weight F32 / activation INT8 | 選択一致23/23、logit最大差0.14515、確率最大差0.01367 |
| 実DeltaNet 132tokens / 1 head | nativeとWasm最大差0、公式BF16との差0.000420 |
| DeltaNet 66＋66 tokens / client-held state | 一括計算との出力・state差0 |
| QKV 128 tokens・256 rows | 約35.1億命令、base+LoRAの4 query合計約3.35 MB |
| QKV 132 tokens・256 rows | 約36.2億命令、4 query合計約3.45 MB |
| 未量子化QKVの公式BF16との差 | 最大0.03125、RMSE約0.00063。累積影響未検証 |
| QKV weight INT8のみ | 最大誤差0.125 |
| QKV activation INT8のみ | 最大誤差0.09375 |

32/64/128/132はQKVの実operandを切り出したtoken prefix長であり、32/64tokensの完全な公式質問を評価した結果ではない。32～132tokens、2/3選択肢の部分検証で、4～7選択肢の全モデル精度、512tokens、Score rubricは未検証。

型付きchoiceは許可選択肢かnullのみを返し、unknown_probability、abstained、有限・正規化済み確率を出す。型安全性の検査は判断goldと別に報告。Score/ordinalのrubric出力はまだ公開APIにしていない。状態checksum/pack違い/重複選択肢の拒否、非ゼロ行タイルの部分読み出し、再送一致、upgrade後の再推論を確認した。

## 制約と次の実装

この時点で全面採用の判断は保留する。617には改善候補がある一方、maximum問題を見逃し、620で順序依存がある。代替案は既存の数値差分ルールで重大変更を必ず検出し、Imajevの判断は説明・曖昧な根拠の補助として扱う構成。原文全体/情報を増やしたpromptでmaximum誤りを調査し、未使用問題を追加して再評価する。2Bへの変更や他モデル採用は今回は行っていない。

全モデルのpack、partial RoPE、層接続、client-held分割と再開は実装済み。原型から768行tileへ拡大した第1層は239→124 query、通信331→171 MBで出力ビット一致を確認した。

残る評価・効率化は、全32層BF16 A/B、複数問題/順序のINT8完走、4〜7選択肢・512tokenの境界、query内の演算融合と入力再送削減、重みINT8と中間状態INT8の独立評価、warm/cold反復。既存のホスト診断ではmaximum変更の見逃しと620の順序依存が残る。全体採用の判断は引き続き保留。


2026-10-03追加（毎回の再処理を除く探索）：prefixをcanister通常queryで一度だけ可逆圧縮し、クライアントが返答packetを保持・再利用する経路を独立診断に実装。全24層・72 queryで元F32状態と全bit一致。毎質問のprefix復元部分は48.6145〜48.6361%減、24層合計2,032,791,617命令減。独立診断の初回準備は6,326,348,500命令・61,219,865 Candid bytes、通信は増える。全推論へ未接続・主問題67 queryのまま。[PREFIX_HYBRID.md](PREFIX_HYBRID.md)。全tokenで固定重み展開を共有する別候補は33 queryでbit一致したが、5実入力で2.28〜3.36%命令増のため不採用。[LINEAR_PAIR.md](LINEAR_PAIR.md)。


2026-10-03追加（50 queryへの探索継続）：共有加算の二段Winogradを実装し、全22通常queryでscalar/旧kernelとbit一致。ただし主Q命令は0.3280%増、prefixは10.7287%増で不採用。[WINOGRAD2.md](WINOGRAD2.md)。標準SIMDの整数内積をWasmで直接生成する候補も全22 queryでbit一致したが、主Qは2.0731%増、5実入力すべて悪化で不採用。[DIRECT_WASM_PAIR.md](DIRECT_WASM_PAIR.md)。87行を静的展開する候補はthinLTO/cgu1とLTOなし/cgu8の両方でコンパイルSIGKILL。実行時性能とは扱わず、明示featureに隔離した。採用済み主canisterの全推論は67 queryのまま、50/32回は未達。

2026-10-03追加（毎回の準備処理を除く実装）：`prepare_prefix_reuse.py`にmodel/pack/input/module/source hashを束ねたクライアント保存とcache hitを実装。専用codecで初回24準備query、2回目の準備query・命令・Candid通信は0。全24 packetは既存独立検証とbyte一致。全推論へは未接続なので主67 queryの削減値とは扱わない。[PREFIX_HYBRID.md](PREFIX_HYBRID.md)。固定係数/入力変換を共有する直接Wasm Strassen1も22 queryで全bit一致したが、主Q+2.5638%・5実入力すべて命令増のため不採用。[WAT_S1.md](WAT_S1.md)。

2026-10-03追加（prefix再利用の本体接続）：実験feature/明示CLIで、opaque packetを型付き入力としてDelta本体へ渡す経路を実装。元logの復元と毎層のclient logコピーを省く。実入力5条件の完全なnative Delta出力/convが旧経路と全bit一致、Rust58/Python13/Wasm check通過。別targetの全Wasmビルド中、新規ローカルcanister6eydd-o3777-77775-aaama-caiは作成済みだが空。全層実測/採用/50回達成は未検証、主67回のまま。[FULL_PREFIX_REUSE.md](FULL_PREFIX_REUSE.md)。
# 2026-10-03: 繰り返す処理の削減を全推論で検証

追加探索：pair-factorを標準SIMDの直接Wasmで実装し、固定operandを全tokenで共有。独立診断の22通常queryで旧pair/scalarとbit一致したが、主Qは1,044,316,378→1,090,457,569命令（+4.4183%）、実5条件すべて悪化で不採用。投影自体も増え、入力準備だけの問題ではない。[WAT_PAIR_FACTOR.md](WAT_PAIR_FACTOR.md)。主canister・全モデル候補・Layaは変更せず、検証済み主suffix64 queryのまま。50/32回は未達。

追加実装：suffix冒頭のembed→norm0→Delta0を通常query内で接続。token ID/conv/可逆prefix packetだけを送り、embeddingとnormの再送を省いた。専用module `89ee3431…` で主・情報不足・最大変更66→64 query、主問題262,163,826,219命令/124,634,637 Candid bytes。直前版より27,272,230命令・1,338,559 bytes減。全5条件のhidden/stateと判断4条件のlogits/probabilitiesが旧INT8版とbit一致、統合Wasmのraw embed/Delta replyも旧Wasmと一致。Rust77/Python16通過、二度目のpacket準備query/命令/通信0。MLP後半＋次Deltaの個別測定合計は5.138 Bで上限を超えるため、単純統合は未採用。詳細は [PREFIX_START.md](PREFIX_START.md)。50/32回は未達、主canister・Layaは変更していない。

追加実装：最終Attention→MLPの往復を統合し67→66 query。さらに復号した2 MiB Delta状態の所有権を再帰演算へ渡し、24層計48 MiB/質問のコピーを省いた。requestをJSONへ二度serializeする照合も全fieldの直接照合へ変更。専用canisterの最終module `c64f1605…` で全5条件のhidden/state、全判断条件のlogits/probabilities/typed outputが旧INT8版とbit一致。主suffix87は262,191,098,449命令、125,973,196 Candid bytes、最大query4,555,649,984命令。コピー削減前の統合候補より52,675,468命令減（−0.02009%）、通信は同じ。統合自体はqueryを1回減らすが命令は微増。Rust全feature74 unit・identity改変integration1件通過、二度目のpacket準備query/命令/通信0。50/32回は未達。[REPEATED_WORK_REMOVAL.md](REPEATED_WORK_REMOVAL.md)。主canisterとLayaは変更していない。

最終候補は量子化入力のduplicated Vecも省き、module `52a7e3f4…`、主87-token suffix262,235,232,497命令（旧採用版比−1.3899%）、67 query。packet無しでは264,169,272,258命令（−0.6626%）、通信増0。全5条件でhidden/stateと全判断条件のlogits/probabilitiesがbit一致、失敗/replay0。観測heap終了値最大4,123,721,728 bytes。新cacheでも二度目のpacket準備query/命令/通信0を確認。`artifacts/prefix_codec/full-direct-input-proof/report.json` / `validated-source.zip`。source内のfeature一覧と再生成kernelが検証済みbuildに一致することも確認し、過去のignored build metadataを再ビルドの前提から除いた。主canister・Layaは変更していない。

固定重みの展開・入力ロードの再利用を増やすWAT候補と、client-held prefix packetを本体へ接続した。専用canister `6eydd-o3777-77775-aaama-cai`、module `44640eb4…`で全5条件のhidden/state、判断を行う4条件のlogits/probabilities/typed outputが旧INT8版と一致。主87-token suffixは265,931,336,448→262,456,310,514命令（−1.3067%）、67 queryのまま。通信113,709,005→125,984,454 bytes（+10.7955%）。同候補の旧log経路では264,360,817,275命令、通信は旧版と同じ。prefix packetの初回24準備queryは次回0になることを実測。詳細は `FULL_PREFIX_REUSE.md`、`PAIR_OPERAND_REUSE.md`。主canisterは変更せず、50 queryは未達。次に量子化入力の複製bufferを省く独立候補を測定する。
