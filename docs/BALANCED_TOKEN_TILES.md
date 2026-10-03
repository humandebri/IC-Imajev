現行の採用結果は [BYTE_BUFFER.md](BYTE_BUFFER.md)。この文書は44＋44kernel単体と全層比較の記録。

# INT8投影の重み再展開を減らす

2026-10-03。主問題のsuffix87 tokenはpadding込み88 tokenを48＋32＋8に分けていた。同じ固定INT8重みを各256列blockで３回展開し、末尾８tokenは一時I32配列を通る旧経路を使っていた。

`experimental-balanced44` はpadding88（入力81〜88 token）を44＋44へ分け、重み展開を２回にする。両方で整数dot直後に元のF32 scaleと加算を行う既存kernelを使う。INT8重み、activation block256、padding、各tokenの積和順序、BF16境界、元F32 LoRA、readoutとcalibrationは維持する。入力長だけで選択し、他の入力長は既存kernelへ渡す。

専用診断canister `5bmyi-tt777-77775-aaahq-cai`、module `b6d2e266d01d9f144e8d6d5870fe25ba3b793e0db4515690252fdd8d48337796` で、layer3 Qの固定重み8192×2560を準備し、実入力５件と符号端点・zero・微小値の境界入力24件を比較した。58通常queryすべてで、両Wasmの出力digestが独立scalarのnative出力と一致した。固定重み準備は65 update、前後module status確認は別に２回行った。

主問題87 tokenの投影は1,157,906,009→1,119,201,385命令、3.34264%減。81〜88 tokenの境界入力は3.34133〜3.35039%減。他の実入力45/80/89/132 tokenは切り替え判定の40命令だけ増えた。これはbase Q投影単体の結果であり、LoRA・全モデル・query数・判断精度の改善を意味しない。counterは入力復元と量子化を含み、digest計算とCDK Candid encode/decodeを除く。

native runtime60 tests、INT8 integration2、F32 matrix integration2、compile-fail doctest3が通過した。INT8比較では81〜88 tokenのすべてと前後の入力長、256/512列、重み-128/127、activation-127/127、異なるscaleを独立scalarとbit比較した。

記録は `artifacts/balanced44/check/report.json`、診断source11 hash/archiveは `diagnostic-source-hashes.json` / `diagnostic-source.zip`、全推論候補source43 hash/archiveは `source-hashes.json` / `validated-source.zip`（すべて同directory）。生成物はgitignore対象。

## 全層比較（44＋44版）

全５条件で保持hidden/state・判断・確率がbit一致、失敗/replay0。query数と通信量は従来MLP full v2と同じ。

| 条件 | query | handler命令 | 旧版との差 | Candid bytes | 実測秒 | 最大query命令 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 90 | 156,358,161,896 | +9,404,920 | 145,902,403 | 20.7158 | 2,699,683,307 |
| 617 | 122 | 287,216,716,220 | -6,365,140,582 | 262,955,428 | 35.8530 | 4,515,331,877 |
| insufficient | 91 | 259,187,850,529 | +19,044,056 | 173,060,522 | 31.4163 | 4,522,464,558 |
| maximum | 122 | 305,939,659,599 | +20,281,880 | 267,767,242 | 37.5833 | 4,767,482,861 |
| normal | 292 | 447,493,422,907 | +17,747,536 | 580,229,781 | 59.0134 | 2,882,723,911 |

主問題は6,365,140,582命令（2.1681%）減。他の条件は0.00397〜0.00735%の微増を記録した。一般的な速度向上を保証しない。共通prefix45＋主suffix87、情報不足45＋80、最大変更45＋89、prefixなし132 tokenは同じ。公式BF16との差と最大変更gold=yesをnoとする既存誤判定は残る。型安全性と判断精度は別に評価する。

固定721 tensorとRoPEの準備は721 update、15,041,482,761命令、258.5515秒。中間状態はクライアント保持、推論は通常query、上限query5B/frame2MB/通常float900K/heap4GiBを維持。最大観測heap4,119,986,176 bytesは旧版と同じで、終端page数の観測であり瞬間ピークではない。通信counterはHTTP/CBOR/signature、命令counterはCDK Candid encode/decodeを除く。時間はcache未制御の単回実測。

module `0fbe5db8c4245bad558aa574ca44fdf92bfa73c2d08863dd74f7fc4d5143f520`、Wasm `artifacts/balanced44/full.wasm`。認証module/cache bookendsと43 source hashは `artifacts/balanced44-v1-cache-checks/report.json`、全層記録 `artifacts/balanced44-v1-*`。archiveと検証source hashの一致を確認済み。

主87 tokenのMLP２query合計は最大5,059,329,973命令になった。１query融合により中間INT8状態の返信・再送・復元を省く候補を続けて検証する。44＋44版では50/32 query未達、主122/初回212 query。

## 87 token融合だけでは実query上限を超えた

87 tokenへ全MLPの範囲を拡張した候補moduleは、prefix３件と情報不足２件でbit一致したが、主問題layer0の最初のqueryでIC0522（single message 5,000,000,000命令超過）となった。採用しない。`artifacts/balanced44-mlp87/failure-report.json`、`partial.log`、`full.wasm`、43 source hash/archiveを保存した。分割handlerの合計はCandid decode/encode等を除くため、融合queryが上限内に収まる根拠として十分ではなかった。通常queryで上限を実測する必要がある。

次の候補は同じ通信型を保ち、query入口の通常Vec復元をCandidのbyte buffer一括復元へ替える。87 token融合を再測定する。候補のビルドやnativeテストだけで完走・精度・query削減を主張しない。
