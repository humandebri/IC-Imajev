# 残差byteの辞書SIMD復号

2026-10-04。50 queryへの連結余裕を作る実験。重み・adapter・readout・BF16境界・token数を変えず、質問状態はクライアントが保持する。既定codecと既存54-query経路は変更しない。

## 実装

carry byte-planeのmode3を追加した。最多15種類のbyteを4-bit番号へ置き換え、番号15は元byteを別の列に置く。これは可逆な通信形式で、INT4量子化ではない。低位byteは従来のraw/Huffmanを維持し、明示的な `residual_dictionary=True` で上位byteにだけ使う。unrounded F32 tail、scale、整数product、LoRA A部分和はそのまま送る。

Wasmでは8 code bytesから16個の辞書番号をSIMDで展開する。例外なしのブロックはswizzle/storeで一括処理し、例外を含むブロックだけ元byteを読んで復号する。重複辞書、範囲外番号、escape欠落、余分なescape、奇数長のpadding、不正なtable/code長を拒否する。圧縮descriptorから出力配列を確保せず、出力長は検証済みshapeだけで決める。

## 復号単体の実測

最初のscalar版は `dictionary-check-v1` の全42条件でHuffmanより約176〜206万命令多く、不採用とした。SIMD版は `dictionary-check-simd-v2` の全42条件（7層×3入力×2つの前半幅）、計84通常queryで保存参照の全byteを含むdigestが一致した。同じ入力・層の残差は二つの幅で同一なので、異なる残差は21組である。

|入力|Huffmanから減った命令/query|増えた要求Candid bytes/query|
|---|---:|---:|
|主87 token|5,143,230〜5,187,734|32,397〜33,956|
|情報不足80 token|4,753,121〜4,803,246|29,857〜31,133|
|最大変更89 token|5,233,167〜5,293,475|33,103〜34,743|

主layer3では10,433,106→5,289,876命令、要求301,683→334,311 bytes、返信92 bytes。観測heap36→38 pages。counterはbyte復号だけを含み、digest計算とCDK Candid decode/encodeを含まない。単回時間にはCLI起動や通信も入り、速度の改善率として使わない。

ベンチはモデルと同じ `carry_planes.rs` を直接コンパイルした。canister `6n3i7-yt777-77775-aaanq-cai` は新規のローカル診断用で、owner以外の状態を保持しない。scalar→SIMDへのreinstallはこの診断canisterだけで行った。Wasm `28bc9d4a3608bf10b86628d655817dfafa592ff0cf8b42b68c450c258ad7e10b`。追加の実queryではSIMDの例外なし/あり2経路のdigest一致と、不正payload6種類の拒否を確認した（`dictionary-build-simd-v2/rejections.json`）。

nativeベンチ5件、Python join5件、既存codec16件/journal6件が通過し、保存MLP payload315件もbyte一致。候補runtimeの128 unit/15 integration/9 doctestが通過（fixture依存1件ignore）。新featureや依存crateは追加していない。

## 推論への組み込み

候補buildは `artifacts/prefix_codec/full-build-residual-dictionary-v1`、四つの既存WAT差し替えとwasmparser検証を通過した。Wasm hash `1190a0e6429fcdefe39c3fc7fcf9f63296f9cdf0534b1bf9f0614305fa72f731`。専用実験canister `6eydd-o3777-77775-aaama-cai` だけをupgradeし固定重みを準備する。主・Layaは変更しない。

単体codecの削減を全推論やquery削減として扱わない。次の判断は87-tokenの26-head連結が5B命令と2MB/frameの両方に収まるかで行う。現時点の検証済み全体は54 queryで、50/32は未達。

辞書だけを組み込んだ `residual-dictionary-chain-main-v1` では24-head Pairが4,801,571,129命令・3,569,645 Candid bytesとなり、直前4,806,737,778から5,166,649命令減、32,628 bytes増だった。保存参照のhidden/norm/carry/convがbit一致。ただし26-head Pairと、24-headの次query（Delta残り8 headとfull MLP）はIC0522で、五連結は未成立。成功metric/profileと失敗request/hashを保存した。

## 並べ替えのSIMD化

profileの `pair_residual_planes` は21,001,293命令で、plane復号後にbyteを交互にpushする処理も負担だった。元のcodecに関係なく、二つの16-byte laneをshuffleして32 bytesずつ書く処理を追加した。decoderのshape/finite/progress検査と出力形式は維持し、tailは従来通りそのままコピーする。

`interleave-check-v3`（最終scriptで再確認した `interleave-check-review-v4`）の42条件・84通常queryで全byteのdigestが一致。主87 tokenは並べ替えだけで10,022,263命令、情報不足80は9,215,863、最大89は10,252,663命令減った。要求/返信の長さは同じ。これは並べ替え単体の値で、全推論の改善率ではない。奇数長・16-byte境界・不正shapeのnative検証も追加した。

次候補buildは `full-build-residual-interleave-v2`、Wasm `2f7fb1f24a7016167102cf2fe9847102df0c0a121d27ab511e5617475f2c18cf`。専用実験canisterだけをupgradeした。辞書と並べ替えを合わせた連結queryを再測定する。

`residual-interleave-chain-main-v2` では主24-head Pairが4,789,488,798命令となった。辞書だけの候補から12,082,331命令減、今回開始時の復元hoist候補から合計17,248,980命令減（0.3588%）。Candid合計3,569,645 bytesは辞書候補と同じ。`pair_residual_planes` は21,001,293→8,918,962命令、その他の主要spanは同じ。成功出力はbit一致、26-head Pairと残り8-head/full MLP queryは依然IC0522。五連結/全体query数削減は未達。

最終候補nativeは129 unit/15 integration/9 doctest通過（1件ignore）。並べ替えベンチのnative6件も通過した。固定重みの再準備は721 update・78,739,929,194命令・252.089秒で、推論の命令・時間とは合算しない。

別方向の調査 `zero-panel-analysis-v1.json` では21組の実activationを確認した。入力整数のzero割合は1.10〜1.53%、product整数は4.59〜7.76%。8要素すべてzeroの割合は入力0、product最大0.001953%。zero-panelを飛ばす方式は分岐負担に比べて削減量が小さい候補として優先度を下げた。疎行列kernelの実測や性能改善として扱わない。

## 最終回帰と次の実装

`full-residual-interleave-proof-v2` の全6条件、`rolled-residual-interleave-regression-v2` の全3条件が完走し、保存対象のhidden/state/final hidden/型付き判断/logits/確率が固定INT8参照とbit一致した。失敗/replayなし、source/moduleを前後確認して79/69ファイルのソースZIPを保存した。compact tailの非返却layer30 hiddenは比較していない。

標準62-query主経路は234,721,319,375命令・123,281,081 Candid bytesで直前と同じ。既存連結経路は主54 query・240,546,093,202命令・133,896,992 bytes・最大4,900,654,494命令・heap観測最大4,144,037,888 bytes・単回37.817秒。直前から892命令増で、今回のcodec最適化は既存graphへ未接続。情報不足は54 query・220,045,107,611命令、最大89 tokenは62-query fallback・240,030,890,467命令。最大変更の見逃しを含め判断精度は変わらない。

次は128行単位のMLP generationを実装する。gate/up projectionは8行単位、downのF32 A継続は8列単位に対応済みだが、streamのshapeとproduct量子化が256行単位に制限する。量子化block256を変えず、未完成の128 productを元のBF16で保持し、次の128と合わせて一度だけ量子化する。入力q/Aとdown A部分和も保持し、演算順序を維持する。6016行へ移すことで直前/次queryの負荷を均す案で、命令上限の成立と最終出力一致は未検証。部分経路が成立しても全体50 queryの証明とは扱わない。
