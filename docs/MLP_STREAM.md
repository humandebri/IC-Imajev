# 再計算しないMLPの途中分割

2026-10-03。50 queryへ計算を詰めるため、MLPをgate/upの出力行境界で止めて再開する通常query APIを実装した。現在の主graphはこのAPIへ未接続で、62 queryのまま。単独でMLPを分割してもquery数は増えるため、性能向上として採用したとは扱わない。

## 一度だけ計算するもの

`mlp_stream_prepare` は残差加算・norm、norm入力のblock256量子化、gate/up両方の元F32 LoRA Aを一度だけ計算する。`mlp_stream_next` はその整数入力・scale・F32 Aをそのまま使う。各product chunkを元の256行境界で一度だけ量子化し、down側の元F32 Aは列の加算順序を保った部分和で継続する。`mlp_stream_finish` は生成済みの整数product・scale・F32 Aでdown projectionと次のnormを完了する。

query間の状態はすべてクライアント保持。INT8の値とscaleを復元して検証するが、再量子化しない。F32 Aと部分和も再計算・量子化しない。質問状態のupdateやcanisterへの永続化はない。

`dims=[tokens,生成済み行数,今回の行数]`、行数は256の倍数、tokensは1〜89。元の残差はBF16、norm/productの整数はI8、scaleとAはF32で保存する。request identity・progress・tensor/norm・有限値・scale・整数範囲を検証したopaque型を、所有権を渡す評価APIで消費する。方向や生成済み行数の取り違えは拒否する。最大89 tokenのcodec状態は約1.59 MBで、2 MB上限内。

## 実測

専用ローカルcanister `6eydd-o3777-77775-aaama-cai`、module `a4a7160898318318c305d3469336c0314a10836513f1b8cf09948a362018f52d`。

実際の主87・情報不足80・最大変更89 tokenのlayer0/1/3について、次の5分割を測定した：2048+7168、3840+5376、4096+5120、4608+4608、2048+2048+5120。全45条件で、生成済みproduct整数/scale/down Aは既存の一括MLP準備とビット一致し、最後のhidden/normは元のMLPとビット一致した。

各入力条件51通常診断query＋9 profile診断query、3条件で180 query。推論全体のquery数とは別計上する。profileは最初のqueryで入力量子化1回・gate/up A各1回を記録し、後続のnext/finishにそれらの実行がないことを確認した。product量子化は生成chunkごとに1回、finishでは0回。

主87・layer0の4608+4608は次の実測だった。frame stepによる微小差は保存reportを参照。

|段階|命令数の代表値|内容|
|---|---:|---|
|prepare|約1.583 B|入力準備と最初の4608行|
|next|約1.504 B|残り4608行、down A継続|
|finish|約1.181 B|down projection、次のnorm|

単独の分割は合計約4.268 B命令・Candid約7.18 MBで、一括MLP約4.053 Bより命令が増える。wireのencode/decodeとcarryのコピーが追加コスト。融合するときはquery内部で中間状態を直接消費し、この余分な返送・復元を省く必要がある。分割単体の速度・通信改善を主張しない。

Rustは候補と同じfeatureでruntime100・canister9 unit、integration9、compile-fail doc7を含めて通過。最大状態のbit roundtrip、進捗の取り違え、NaN・不正整数・trailing bytes・request変更の拒否も検証した。

## 既存graphの回帰と未達事項

同じ候補で全6条件が完走し、返却hidden・保持state・型付き判断・logits・確率が既存INT8版とビット一致した。非返却layer30 hiddenを直接比較したとは扱わない。失敗とreplayは0。既存の最大変更の見逃しも残る。

主87は62 query、235,201,572,540命令、Candid 123,281,081 bytes、最大4,716,852,888命令、単回35.046秒。API分岐の追加で直前より8,528命令増、query数・通信は同じ。cold132は既存経路290 queryで359,797,102,290命令。現在のgraphを50/32 queryにしたという証拠にはならない。

次にDeltaのhead群を処理しながらout projectionのINT8 block部分和と元F32 Aを保持する必要がある。MLPの後半とDelta前半を同じqueryに置き、続くqueryでDeltaの残りと次層MLPを置く構成を検証する。命令counterの足し算だけでquery数や通信上限を満たしたと判断しない。

固定モデル準備は721 update・4,065,416,192 bytes・78,739,929,194命令で推論と別。主canisterとLayaを変更せず、mainnet・push・PR・commitを行っていない。生成物はignore済み。

証拠は `artifacts/f32_k_continue/stream-check-{617,insufficient,maximum}/report.json` とsource ZIP、全体は `artifacts/prefix_codec/full-mlp-stream-proof/report.json`・`before-after.json`。ビルドは `full-build-mlp-stream-v2` にソース、feature、kernel、compiler、bridgeのhashを保存した。`scripts/check_mlp_stream.py` と `scripts/archive_mlp_stream_proof.py` で実測と証拠検査を再実行できる。
