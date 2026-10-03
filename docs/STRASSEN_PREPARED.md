# 固定Strassen係数の準備と繰り返しアドレス計算の除去

2026-10-02。固定INT8重みから導出する7行列のI16展開・補正復元・元重みとの全要素照合を、毎queryから準備updateへ移した。`PreparedStrassen`の係数・scale・model/pack識別子は変更不能で、queryは入力の量子化と積和を行う。元INT8、block256のactivation scale、F32積和順、BF16境界を保つ。質問に依存する中間状態は保存しない。

`experimental-strassen-prepared`は既存wide実験を含み、`experimental-strassen-packed-reduction`はpreparedをcanister/runtime双方で含む。既定の通常INT8経路には有効化しない。`warm_weights(source+".strassen_i8")`が、sealed packの範囲とcache予算を検査して係数を準備する。元INT8の4象限から各導出係数を独立に確認し、scaleの正の有限性も検証する。prepared評価は固定識別子、shape、入力・出力・work上限を引き続き検査する。

診断対象はlayer 3 Qの8192×2560投影。実際の全層実行から保存した1/7/8/32/64/87 token入力と、別のprefix/情報不足/最大変更実行の45/80/89 token入力を用いた。各候補で9形状×2演算×cold/warm/clearの54通常queryが完了し、3候補合計162queryの出力が通常INT8・独立したnative scalarとbit一致した。これはbase Q投影の検証であり、全モデルの判断精度やquery削減の検証ではない。

| tokens | 通常INT8 warm | prepared | SIMD packed reduction | 行アドレス共有を追加 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29,929,927 | 179,200,274 | 171,888,985 | 155,047,246 |
| 7 | 142,483,851 | 194,861,270 | 187,549,981 | 170,708,242 |
| 8 | 143,759,745 | 197,491,436 | 190,180,147 | 173,338,408 |
| 32 | 528,368,236 | 866,217,204 | 823,711,035 | 573,458,736 |
| 45 | 780,709,370 | 1,195,897,410 | 1,141,635,721 | 850,172,542 |
| 64 | 1,040,832,364 | 1,610,089,364 | 1,537,979,355 | 1,206,889,424 |
| 80 | 1,301,931,804 | 1,947,629,284 | 1,863,763,755 | 1,491,462,944 |
| 87 | 1,435,687,374 | 2,119,862,934 | 2,028,686,045 | 1,639,543,506 |
| 89 | 1,541,800,178 | 2,435,388,570 | 2,320,772,321 | 1,739,430,102 |

87 tokenはprepared21.20億→SIMD packed20.29億→行アドレス共有16.40億命令へ低下したが、通常INT8の14.36億より14.20%多い。全9形状で通常INT8に勝てず、不採用とする。cold/clearでも87 tokenは通常1,477,258,507に対し候補2,244,559,378命令。handler計測はCandid decode/encodeを含まない。

行ごとのchecked offsetをSIMDロードのたびに計算していたため、dot入口で行アドレスを一度作り、列offsetだけ加える形に変更した。全体の範囲・形状検査と整数overflow検査設定は維持する。R32 leafの生成Wasmは15,206→10,620命令、BrIf469→5、I32Add750→10。これは静的コードの個数であり、queryの動的命令数は上表で評価した。

係数準備は73,433,088 bytes、通常baseとのcache合計94,437,376 bytes。最初のprepared候補で係数準備updateは5,337,475,044 handler命令・0.8313秒・Candid request73/reply178 bytes、base準備は55,254,358命令。重複準備は4,191命令でcache増加なし。clear2回・準備3回の5updateを推論queryとは別に記録した。warmの両演算はphysical stable read 0、cold/clearは通常21,004,288 bytes、候補38,278,204 bytes。warm終端heapは2,971 pages（194,707,456 bytes）で、割当peakの測定ではない。全denseを同方式でI16へ展開すると元INT8係数の約3.5倍となり、現在の4GiB heap/cacheへ全モデル分を保持できない。132 token×全8192行も900,000出力要素上限を超え、この部分候補は非ゼロrow tileに未対応。

初期packed buildではroot featureからcanister側prepared featureへの依存が欠けていた。coldの18queryは一致したが準備で拒否され、完了reportは存在しない。依存を修正しcompile_errorによるguardを追加した後、packed-fixed54queryとaddress54queryを改めて完走した。失敗した初期trialは成功件数に含めない。通常Rust43＋cache3＋compile-fail doctest1、prepared/packed Rust44＋cache3＋doctest1が通過した。

再現は`check_prepared_strassen.py --canister <diagnostic> --wasm <candidate> --directory <new-output>`。対象は専用の59,249,724-byte部分pack（SHA256 `0aa00943520728ed0a610e0120402466e8d9778b4fb7f6d516aec55a1f2ab9b6`）をsealed済みの診断canisterに限る。full packへそのまま適用しない。元モデルrevision・公式実装・tokenizer/readout/calibrationは変更していない。

生reportは`artifacts/strassen-prepared/{probe,packed-fixed-probe,address-probe}/report.json`、Wasm audit、source hash、失敗ログ、復帰確認は同ディレクトリに保存しgitignoreした。候補module SHA256はprepared `0d9cec88a1ab8b75e10c52493dfdb4a6f57e83c5b25ddeec34724ddc58340ecc`、packed-fixed `9e91d6f7633dc500cea0451a282fc21e3abd09fa393ef4af7a133b1c8d92ea30`、address `67423b9d8c6b795dcb24619b2347e6fd5ac9cd8c8b20543bfc162839a9e3be27`。

試験後、診断canisterを元module `7c366d0b7958a3034b89dff98c0fe1486239d3419cad4882324231c7268b59b5`へupgradeし、認証済みhashで復帰を確認した。主canisterは`86d93cda02bb5bb6c6e4d17bea38b3315206aace76cdc94d3a4941a6edbec0df`のままで、721 tensor・4,065,416,192-byte cacheとfull packを再照合した。共通targetのWasmもこの採用済みartifactへ戻した。Strassen試験終了時のソースからの通常feature rebuild `79bf2b709c5ba6f45e88fbb73d814d676b2dbc79dafa3270703f06c436fba866`は別artifactに保存し、未deploy・全層未検証である。主canisterの既存全層実測をこのrebuildの実測として扱わない。Layaは変更していない。

繰り返し処理の残る候補として、同一入力のrow分割を採用版の保存requestから調べた。prefix準備済み305queryでは対象142投影groupに重複row呼び出し0、prefixなし485queryでは197groupのうち62groupで同一入力を2回送っていた。後者は量子化・norm・LoRA A等を分割間で共有する対象だが、通常queryは変更を永続化しないため、再利用するならcanisterで生成した中間表現をclientが保持して次queryへ渡す必要がある。まだ実装・精度検証・通信費用の評価をしておらず、削減には計上しない。hotの最大対象は引き続き整数dotとLoRA積和である。
