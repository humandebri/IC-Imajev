# Delta出力投影の列継続

2026-10-03。分割queryで既処理の列を再量子化・再計算しないため、Deltaのout projectionを256列単位で継続する部品を実装した。中間状態はクライアントが保持し、固定重みの準備以外は通常queryで進める。

`linear_integer_k_continue` は今回のBF16 gated入力だけを量子化し、元のblock256順序でF32 base部分和に加算する。元F32 LoRA Aは `matmul_k_continue` で継続する。`linear_integer_k_finish` は保持したbase/Aから元F32 Bだけを計算し、元のBF16丸め順序で完了する。途中のF32部分和を量子化しない。

実入力45/80/87/89 token、layer0/30の8箇所を、3072+1024、2816+1280、2048+2048、1024×4で検証した。全32条件でbase・A・最終LoRA出力が、一括計算および保存済み既存INT8版とビット一致した。216通常診断query、3 profile query、6拒否queryは推論全体とは別計上する。初期−0、BF16以外の入力、非整列列境界、誤った状態サイズ/alpha/adapterの拒否も検証した。

主87/layer0の2816+1280 profileでは、新入力の量子化は各chunkで1回、finishでは0回。finishはBを1回計算し、Aを再計算しない。単独の分割は5 query・670,388,095命令・Candid 5,986,593 bytesで、一括projectionよりqueryと通信が増える。MLP後半とDelta前半、Delta後半と次MLPを融合するための部品として扱い、単体の性能改善として採用しない。

候補module `56504549d2bf73f74a66fcc033369ba70870d30d8ed8e4ef65d692203b96e891` を実験canister `6eydd-o3777-77775-aaama-cai` で検証した。Rustはruntime103/canister9 unit、integration9、compile-fail doc8を通過。固定準備は721 update・4,065,416,192 bytes・78,748,249,194命令で推論と別。

既存graph全6条件の返却hidden・保持state・型付き判断・logits・確率はビット一致、失敗/replay0。非返却layer30 hiddenの直接比較は含まない。主87は62 query、235,202,813,732命令、Candid 123,281,081 bytes、最大4,716,872,899命令、単回34.676秒。分岐追加によって直前より1,241,192命令増えた。通信/query数は同じ、観測終了heapは+65,536 bytes。既存の最大変更見逃しは残る。50/32 queryは未達。

証拠は `artifacts/f32_k_continue/int8-column-check/report.json`、全体は `artifacts/prefix_codec/full-int8-column-proof/report.json` と `before-after.json`、対応ソースZIP。buildは `full-build-int8-column-v2` に固定した。`scripts/check_int8_column_continue.py` と `scripts/archive_int8_column_proof.py` で再測定・証拠検査する。主canisterとLayaは変更せず、mainnet/push/PR/commitは行っていない。生成物はignore済み。

## 未使用列の変換を省く探索（2026-10-04）

列部分和は過去のdotを実行しないが、S1の入力係数は全4096列を変換していた。新しい列だけを変換するv1候補を実装し、32条件の出力一致を確認した。しかしfull stride bufferのゼロ埋めを追加したため、主87の2分割は約279万命令、4分割は約334万命令増えた。module `bc7b3ec7d6593dbdb185ad950a5888bf66ff085c14c4224dd889df2075e964da` は未採用。証拠は `artifacts/f32_k_continue/int8-range-check/report.json`。

追加のゼロ埋めを避け、必要な範囲だけ初期化して読むv2は、実canisterの全32条件で出力bit一致・命令減・通信不変だった。主87/layer0の2分割は225万命令（約0.336%）、4分割は675万命令（約0.913%）減。元の全投影経路は維持する。単体試験では選択範囲を全変換と比較し、非選択範囲を読まず、列範囲・overflow・odd token paddingも検証する。

厳密なゼロblockの省略も調査した。検証済みMLP productの主87/情報不足80/最大変更89、layer0/1/3の27,648 token-blockで、256要素がすべてゼロのblockは0だった。個々の整数ゼロは約6.99〜9.11%あるが、block単位のdot省略はこの9入力に効果がないため実装しない。`scripts/check_zero_product_blocks.py` と `artifacts/f32_k_continue/zero-block-exploration.json` に入力hashと結果を保存した。

## Deltaの入力準備を一度だけ共有する診断

`delta_project_capture/reuse` を16+6+10 headへ適用し、45/80/87/89 token・layer0/1/30の12条件を測定した。gated出力全4096列とconv最終履歴は、保存済み既存INT8版とbit一致。最終recurrence stateはkeep=0で返却せず、比較範囲に含めない。入力のstate/historyは元のcanister生成frameから切り出し、クライアントでrecurrenceを計算していない。

主87/layer0の3診断queryは1,453,637,266・517,101,397・854,968,712命令、合計Candid 4,889,123 bytes。out projectionはこの3 queryに含まない。profileで最初のactivation量子化1回・QKV/ZのF32 A各1回、後続2回は量子化/A再計算0を確認した。3 profileを含め39診断queryで、全推論のquery数とは別。

これをMLPと融合して50 queryを達成したとは扱わない。次は一つのquery内部で検証済み中間状態を直接渡し、encode/decodeと連結コピーを省く必要がある。分離queryのcounter加算だけでは、融合後の命令/2 MB通信/heap上限を満たす証拠にならない。診断コードは `scripts/check_delta_head_reuse.py`、証拠は `artifacts/f32_k_continue/delta-head-reuse/report.json` と対応ソースZIP。

## 修正版の全体回帰（2026-10-04）

module `7203222a28819b1a1d0d6fc7325a3bac6859c6929f45fdd44cea7f55f60ee35f` で全6条件が完走し、返却hidden・保持state・型付き判断・logits・確率が既存INT8版とbit一致。失敗/replay0。非返却layer30 hiddenは直接比較せず、最大変更の見逃しも残る。Rustはruntime104/canister9 unit、integration9、compile-fail doc8を通過した。

主87は62 query、235,201,573,732命令、Candid 123,281,081 bytes、最大4,716,852,899命令、観測終了heap最大4,130,144,256 bytes、単回35.201秒。列継続APIへ既存graphは未接続のままで、直前のcolumn-v2から1,240,000命令減は再コンパイルによる差も含む。MLP-stream段階からは1,192命令増。区間準備の225/675万命令減を全推論へ適用したとは扱わない。column-v2の単回34.676秒より時間は増えており、速度向上は未確認。50/32 query未達。

固定準備は721 update、4,065,416,192 bytes、78,739,929,194命令、308.312秒。質問状態の更新はない。生成物はignore済み。主canisterとLayaは変更していない。buildは `artifacts/prefix_codec/full-build-int8-range-v2`、全体証拠は `full-int8-range-v2-proof`、32条件の診断は `artifacts/f32_k_continue/int8-range-v2-check`。`scripts/archive_int8_range_proof.py` が全216診断responseの変更前後bit一致、全体6条件、Delta12条件の準備回数、hash/ソースZIPを検査して保存する。
