# BOOM DAO実例の選定記録

#660 Motion例を#584に差し替えた。#617と#653は既存例を維持する。

## 出典と確度

- #584原文: https://dashboard.internetcomputer.org/sns/xjngq-yaaaa-aaaaq-aabha-cai/proposal/584
- 公開API保存: frontend/data/boom-584-proposal.json
- #586原文: https://dashboard.internetcomputer.org/sns/xjngq-yaaaa-aaaaq-aabha-cai/proposal/586
- 事件の公開技術分析を引用するフォーラム: https://forum.dfinity.org/t/reviving-the-sns-framework-addressing-current-challenges-and-exploring-solutions/59196
- 当時の報道: https://ledgerlife.io/token-trouble-how-a-governance-exploit-rocked-the-icp-network/

公開提案スナップショットでは#584、#586、#601は同一proposer neuron（a18d3c619d686219da2edfe505c3d698af1925c1db8f5cd95dcc8b84613f04c5）。この一致は同一neuronからの提出を示し、本人の身元や悪意を証明するものではない。

#584はSNS Metadata Adjustmentというタイトル・summaryなのに、payloadはSNS財庫から20M BOOMを指定口座へ送る要求。REJECTEDなので成功した財庫流出ではない。説明とactionの食い違いは公開データから直接確認できる。敵対的な提案の検出を考える有力な候補だが、調査で取得できた資料だけから「確認済み攻撃」とはラベル付けしない。

#586は出金失敗を説明するが、最低stake5→1M BOOM、最低投票ロック2日→1461日、wait-for-quiet延長86400→1秒を要求。transaction_fee_e8sはNone（変更要求なし）。EXECUTED。これは別候補として確認したが、#617と投票参加制限の評価が重複するため採用しない。

#617→#653の連鎖は公開技術分析で具体的に説明されている。#617による約55年への投票ロック条件上昇のあと、約20分の窓に作られた#653では2neuronだけが投票資格を満たした。技術的なbugによる採択ではなく、連続する提案と投票資格のスナップショットの効果として分析されている。現在のneuron/ledger試算を当時の状態と混同しない。

## 評価設計

#584: Titleとpayloadの両方を入力に残し、「実行すれば何が起きるか」を問う。metadata update / treasury transfer / no change。正答は財庫送金で、attackという結論を前処理で入力に添付しない。

単一の選定例に正答しても汎用的な攻撃検出能力の証明にはならない。次の対照評価では、同じtitleで正当なmetadata actionを持つ例、異なるtitleの同じtransfer action、本文とactionが一致する資金移動を含める必要がある。今回の公開デモでは原文・質問・選択肢・実応答を保持して再現可能にする。

## 本番モデル実測

2026-10-09、prefix27の公開canisterへの匿名query。#584原文から機械的に作った86tokensの入力でtreasury transfer 62.1%、metadata update 24%、no change 9.7%、unknown 4.1%。実行時間75.3秒。update呼び出し・画面エラーなし。記録: artifacts/boom-584-action-mismatch-20261009/report.json。

合成対照例も同じ公開モデルで実測した。同じmetadataタイトルでロゴ変更だけのpayloadはmetadata update 93%、一般的なSNS Adjustmentタイトルで同じ20M送金payloadはtreasury transfer 85.6%。いずれも匿名queryのみで、update呼び出し・画面エラーなし。対照例の記録: artifacts/boom-584-action-mismatch-20261009/controls/report.json。3入力の結果は説明と実行内容を切り分けるこのデモの動作を示すが、一般的な攻撃検出精度の評価ではない。


## #617 の質問変更とローカル検証

質問を `Does this change indicate malicious governance manipulation?`、選択肢を `likely malicious` / `legitimate change` に変更。情報不足はモデル既定の unknown を使う。現在の neuron への試算であり当時の復元ではないことと、再ロックしない前提をモデル入力内にも明記した。UI は疑いの評価と表示し、悪意の証明とは扱わない。

localhost:8001 の既存モデル（module hash af806025702699f562d6d2eb841155502d05983efedc0da3f2c7f2cbc4918ac5）で新規 query を実行。UI と完全一致する 83-token 入力は likely malicious 67.9%、選択肢を逆順にした入力は同ラベル 71.0%。記録: artifacts/boom617-malicious-validation-20261009/report.json。

初回の合成対照例は正当な変更 85.3%、情報不足 99.9%。一方、乗っ取りリスクを問う別質問では同じ試算に low takeover risk 51.1% が選ばれた。質問への依存が残るため、汎用的な攻撃検出精度は未検証。初回記録: artifacts/boom617-malicious-local-20261009/report.json。

サンプル読み込み・結果表示・実行中編集時の判定表示保持と通常サンプルへの切り替え・モバイル表示を含む Playwright 28 件、データ生成の Python 8 件、ブラウザー tokenizer の上限検査、TypeScript 型チェック、Vite build が成功。本番への推論要求・デプロイは行っていない。
