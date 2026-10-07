# BOOM DAO proposal 500〜660：元proposalと数値参加判定の照合

2026-10-07。approve 2件・reject 10件に加え、モデルhold 38件も全件確認した。モデル対象50件・18種類の入力、およびTool holdの代表・証拠不足ケース7件を確認した。参照したproposalは2026-10-06取得の固定snapshotで、最新chain状態の評価ではない。

数値上の参加条件という限定された方針では、approve 2件とreject 10件の方向は支持できる。一方、hold 38件のうち22件は大幅なstake/投票lockの制限強化を含み、同じ方針での私のレビューはreject相当とする。残り16件は最低投票lock 1→2日であり、悪影響を断定する追加根拠がないためholdを支持する。

これはこのエージェントによる条件付きレビューで、人間が確認した正解ラベルではない。正答率・実際の採否・本番投票推奨は確定していない。過去GPTラベルや採決結果を正解に使わず、現行推論結果も変更していない。

## 元データと入力の照合

全50件についてsnapshot hashを照合し、ハーネスとは別のparserでCurrent/Newの整数・bool値を抽出した。構造化payloadの指定値とNew renderingは一致し、旧値との差分はtaskの変更項目集合と完全一致した。数値が同じ指定値は変更から除外されている。旧値は履歴rendering由来であり、chain上の当時の状態との独立照合はしていない。

圧縮は変更された旧新数値を保持している。元の推論検証に加え、今回全50件の元payloadとの差分を照合した。ただし数値保持は意味理解の保証ではない。長い入力ほどunit_groupedへ切り替えるため、項目数・内容・圧縮形式・質問の短縮が同時に変わる。現データだけで圧縮が失敗の原因だとは断定できない。

数値入力の重複排除では、title/summaryが代表taskと異なるproposalも同じ結果を共有する。今回の該当IDは[601]。review.jsonには各proposal自身の原文を残した。数値判定では同一入力だが、目的や意図まで同一とは扱えない。

最低stakeはneuronの最低stake条件、最低投票lockは投票資格に必要なdissolve delay。最大lockは最大値とbonus計算に関わる値で、全員を自動的にその期間lockする意味ではない。bonusは投票力配分に関わる。[公式SNS設定仕様](https://docs.internetcomputer.org/references/sns-settings/)で用語を確認した。既存neuronへの実際の影響・保有分布・価格・DAO防衛との利益衡量は今回確認していない。

## 18種類の入力ごとのレビュー

|proposal ID|現在の判定|A/Bの方向・未校正スコア|数値参加方針でのレビュー|根拠|
|---|---|---|---|---|
|505|reject|reject / 0.8699|reject|最低stake 5→100万tokens、最低投票lock 2→365.25日。参加条件を大幅に厳しくする。|
|552|reject|reject / 0.8145|reject|最低stake 5→100万tokens。bonus増加は新規参加のstake条件を緩めない。|
|553|reject|reject / 0.6996|reject|最低stake 5→500万tokens。新規参加の必要stakeが100万倍。|
|554|reject|reject / 0.7780|reject|最低stake 5→1,000万tokens。新規参加の必要stakeが200万倍。|
|555,556|reject|reject / 0.7103|reject|最低stake 5→1,500万tokens。新規参加の必要stakeが300万倍。|
|559,563,567|reject|reject / 0.7529|reject|最低投票lock 2→1日は緩和だが、最低stake 5→1,500万tokensの障壁が残る。|
|562,566|hold|approve / 0.5354|reject|最低stake 5→100万tokens、最低投票lock 2→1,461日。両方の参加条件が大幅に悪化。|
|570,571,574,577,580|hold|reject / 0.5883|reject|最低stake 5→1,500万tokens。投票lockの半減はstake障壁を解消しない。|
|586|hold|reject / 0.5344|reject|最低stake 5→100万tokens、最低投票lock 2→1,461日。reject costも100→1万tokens。|
|587|hold|approve / 0.5165|reject|最低stake 5→100万tokens、最低投票lock 2→1,461日。bonusの変更では参加条件を緩められない。|
|588|hold|approve / 0.5036|reject|最低stake 5→200万tokens、最低投票lock 2→1,461日。|
|590,591,592,593|hold|reject / 0.5498|reject|最低stakeは1,500万→600万tokensへ下がるが、最低投票lockは2→1,461日へ上がる。|
|594,595,596,597,601|hold|reject / 0.5176|reject|最低stake 5→600万tokens、最低投票lock 2→1,461日。|
|598,599,600|hold|reject / 0.5207|reject|最低stakeは1,000万→600万tokensへ下がるが、最低投票lockは2→1,461日へ上がる。|
|602,605,608,611,614,620,623,626,629,632,635,638,641,644,647,650|hold|reject / 0.5385|hold|最低投票lock 1→2日の強化だけ。2日を禁止的とする基準や実際の影響は示されていない。|
|617|reject|reject / 0.6865|reject|最低投票lock 1→2万日（約54.8年）。最大lock延長と最低投票条件を混同しない。|
|656|approve|approve / 0.8763|approve|最低stake 1,500万→5tokens、投票期間1→3日。数値上は参加条件を改善。ただしsummaryのtakeover言及は入力から落ちている。|
|659|approve|approve / 0.8625|approve|最低stake 1,500万→5tokens、最低投票lock 2→1日。lock bonus 1→100%の投票力配分への影響は別途検討が必要。|

## approveの限界

#656はstakeを1,500万→5tokensへ戻し、投票期間を1→3日へ延ばすため、数値参加条件のapproveには根拠がある。ただし元summaryには「The takeover can still happen」という言及がある。これは攻撃の立証ではないが、意図・支配権の評価を要する文脈で、数値promptには残っていない。全proposalの承認へ拡張するならholdとして追加検討すべきである。

#659もstakeと最低lockを下げる。一方、lock bonusを1→100%へ変えるので、投票力配分の影響は参加資格の改善だけでは評価できない。両件とも「数値上の参加条件を改善」というラベルとして扱う。

## holdの原因

22件のモデルholdは入力に大幅な制限強化がある。#562/#566はstake 5→100万tokens、最低lock 2→1,461日なのに、A/Bの方向がapproveでスコア0.5354だった。#587/#588もapprove方向だった。低スコアholdと承認証拠gateが誤ったapproveの出力を止めているが、モデル自身が一貫して制限を識別したとは言えない。

#602等16件は最低lock 1→2日だけで、A/Bはreject方向0.5385。要件の強化と「禁止的な障壁」は同義ではない。適切な許容基準がない状況でholdを外すのは妥当でない。

|Tool holdのID|元proposalの内容|レビュー|
|---|---|---|
|584|SNS treasuryから2,000万tokens送金。タイトルはSNS Metadata Adjustment。|送金額は確認できるが、残高・受取人支配・用途の根拠不足。holdを支持。|
|653|2億5,000万tokens mint。|供給量・受取人の保有と支配の根拠不足。holdを支持。|
|654 / 655|8,900 ICP / 1,800万SNS tokensを同じprincipalへ送金。|関係と用途の根拠不足。holdを支持。|
|658|frontend batch 44のcommit。構造化payloadは空配列、renderingは非空payload hashとbatch evidenceを記す。|データ源の不一致とコード未検証によりholdを支持。空配列が実際のon-chain payloadだったとは断定しない。|
|585|stake 5→100万tokens、最低lock 2→1,461日を含む。全変更を保持した入力130tokens。|内容上は制限強化だが、実行予算超過のholdを維持。長さの問題を証拠不足と区別する。|
|501|ledgerのtransfer_fee 100万e8s指定。|数値参加判定の対象外。内容を評価済みという意味ではない。|

## 既存logitsによる閾値比較

追加推論なしで、同じA/B logitsと現行承認証拠gateを使った。下表の比較対象は今回の条件付きレビューであり、人間goldに対する精度や校正結果ではない。

|閾値|approve|reject|hold|レビューreject 32件のうちrejectを返す件数|レビューhold 16件をrejectにする件数|
|---:|---:|---:|---:|---:|---:|
|0.5|2|44|4|28|16|
|0.52|2|39|9|23|16|
|0.54|2|19|29|19|0|
|0.55|2|15|33|15|0|
|0.6|2|10|38|10|0|
|0.7|2|8|40|8|0|
|0.8|2|2|46|2|0|
|0.9|0|0|50|0|0|

閾値0.6ではレビューreject 32件中10件をrejectにできる。0.5では28件になるが、最低lock 1→2日の16件もrejectになる。さらに制限強化4件はA/Bがapprove方向なので、閾値だけではrejectへ直せない。全体の閾値を下げる解決は採用しない。

## 次の改善に使う固定ケース

review.jsonに全50件の旧新値・現行入力・入力形式・根拠・条件付きラベルとsource hashを保存した。内容と形式の影響を分離する比較では、#562（両要件悪化）、#590（stake緩和とlock悪化）、#570（長めのratio）、#602（小幅悪化）、#656/#659（緩和）を優先する。短い単一要件の#553も対照にする。

表現比較は、質問・選択肢・モデル・閾値を固定し、全変更を残したまま資格条件を先に置く表現と現行表現を比較する。128tokensを超えるケースは除外理由を記録し、強制的な86token化や変更項目削除をしない。今回の7ケースで調整した後は別の未使用ケースで検証し、既知ケースへの合わせ込みと区別する。

数値差分と単位はコードで厳密に処理し、参加条件の改善と全proposalの安全性を別項目として記録する設計が必要である。「禁止的」の許容基準を明示せず、増加ならすべてrejectという規則を追加してモデル改善と数えるべきではない。今回は推論・本番policy・元評価ファイルを変更していない。

