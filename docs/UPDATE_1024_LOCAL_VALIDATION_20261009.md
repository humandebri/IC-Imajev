# 1024-token updateのローカル実モデル検証

2026-10-09。入力全体（共通prefix 5 tokensを含む）の上限を1024へ拡張した。外部updateは1回で、内部workerに分けて全32層を計算する。

## 変更

- chunks有効時のsequence位置上限は1024、その他のruntimeは512。tile行数上限は変更していない。receiptの入力記録上限も1024に揃えた。
- 長いAttention履歴では4 query headsずつ処理し、配列900,000 floats・GQA演算量75,000,000の上限を守る。512以下では従来の8 headsを維持する。
- suffixが507を超える入力のworker上限を128へ拡張した。
- 最適化builderの凍結runtimeにも同じ位置上限を適用した。34個の投影kernel本体は保存済みdonorと同一hash。
- 検証用のローカル呼び出しhelperは同一ingressを最大30分pollする。時間切れによる再送はしない。

## 検証対象

共有ローカルnetworkのlocalhost:8001で、専用canisterを新規作成した。検証後にこの3つを停止し、`cleanup.json`へ停止状態を保存した。既存のモデルcanisterと公開環境は変更していない。

- model: `2vn5i-k3777-77775-aaaua-cai`
- relay: `2sm34-hd777-77775-aaauq-cai`
- codec: `23pqa-rl777-77775-aaava-cai`
- 計測用diagnostic WASM: `d4173e45926faebf3dc72d5783907d315e616076743a90ba1f7aaa617248f51c`
- 通常ビルド: `bc2e1cd240f4481260a814b268da4481483f56dd7ac57c90f2ab168aed690900`

実モデルfull-int8.pack 4,702,451,200 bytesをuploadし、canister側SHA照合を通過した。721個のweight cache tensor（4,065,416,192 bytes）を準備した。共通prefix 5の24 Delta cacheと全32層のprefixを登録した。

計測対象は作業開始時に凍結した料金APIで、Configには`enabled`がある。同時進行のAPI整理後の通常ビルドにはこのfieldがない。計測ではCandid helperと呼び出しhelperを保存してABIを固定した。通常ビルドの生成・export検査は通過したが、現時点で実モデル計測対象はdiagnostic WASMのみ。

## 自動検査

- runtime: 75 passed。1024位置で4-head GQAとscalar head計算の出力をbit単位で照合。
- canister: 26 passed。
- chunks無効の有料update互換検査: 24 passed。
- Pythonの有料prefix検査: 6 passed、builder prefix契約: 2 passed。

## 実モデル測定

全ケース完了、`complete:true`。生データは`artifacts/paid-1024-20261009/proof-v1/report.json`。

| 入力tokens | 結果 | worker | 外部update経過秒 | 最大worker命令 | 最大heap bytes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | 全32層完走・同request ID追加料金なし | 75 | 856.421 | 33,990,837,930 | 4,245,749,760 |
| 1025 | 料金受領前に拒否 | — | — | — | — |
| 512 | 全32層完走・同request ID追加料金なし | 36 | 401.509 | 33,272,144,743 | 4,245,749,760 |
| 94 | 全32層完走・同request ID追加料金なし | 7 | 58.261 | 33,367,734,826 | 4,245,749,760 |
| 95 | 全32層完走・同request ID追加料金なし | 7 | 52.725 | 30,929,263,736 | 4,245,749,760 |

1024の総worker命令数は2,346,489,042,974。最大heapは3.95416 GiB、4 GiBまで46.9375 MiB。各workerは40B命令以内。時間は共有ローカル環境のpoll待ちを含み、本番性能の保証ではない。

512→94→95の実行は1024の後なので、heap_pagesは先行実行の高水位を引き継ぐ。

意図的なworker trapと返金失敗で`Pending`を作り、retryで`Done`へ移行した。再retryの二重返金なし、失敗receipt replayの新規料金なし、新しいrequest IDでの復旧後の判定一致を確認した。

実モデルで測定した長さは94・95・512・1024・1025。513・768はfixtureとnative境界検査のみで、このレポートでは実モデル完走を主張しない。

fixtureはBOOM #653の入力にpaddingを加え、prefixとchat末尾を保持した長さ試験。自然言語の精度を検証するものではない。全32層の独立query参照とのlogit一致と本番subnetのcycles消費は未測定。
