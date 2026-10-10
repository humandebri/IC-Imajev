# IC-Imajev 判断プレイグラウンド

ブラウザから公開canister `xis3j-paaaa-aaaai-axumq-cai` に匿名queryを送り、実推論の結果を表示します。tokenizerとqueryの組み立ては同じWeb Workerで実行します。Cloudflareは静的ファイルを配信します。

## 起動とビルド

Node.js 24以上、npm、リポジトリの `.venv/bin/python` とNumPyが必要です。`MODEL_LOCK.json` のtokenizer・readoutファイル、`checkpoints/full-int8.manifest.json`、検証済みprefix archive `artifacts/query-packing-v3/prefix-v2/` を準備してください。モデル重み本体はfrontendのビルドに不要です。

```sh
cd frontend
npm ci
npm run dev
```

http://127.0.0.1:5173 を開きます。`dev`、`build`、`test` の前処理で固定tokenizerのhashとprefix archiveのidentity・各ファイルのhashを検証し、静的素材を生成します。tokenizerは約12.8MB、prefix素材は3,816,448 bytesです。`public/tokenizer/` と `public/inference/` はGit対象外で、ビルド成果物に含まれます。

```sh
npm run typecheck
npm run build
npm run preview
npm test
```

PlaywrightのChromiumがない場合は `npx playwright install chromium` が必要です。

## 入力と実行

質問は必須、判断材料は任意、選択肢は2〜7件です。空欄、前後空白を除いた重複、判断保留用の予約語 `__unknown__`、1選択肢128 UTF-8 bytes超は拒否します。固定5-token prefixと追加1〜91 token、合計6〜96 tokenが対象です。prompt、選択肢、unknown、chat templateを含む実token数で実行可否を判断します。

`Run inference` で送信時の入力を固定し、token数に対応する測定済み計画で32〜63回の推論queryをcarryの依存順に呼びます。計画は現在のmodule hashに固定し、MLP生成を256単位で分割します。実行前に2回のreadiness query、実行前後にmodule hashの証明を確認します。結果には選択値、候補別確率、unknown、判断保留を表示します。途中で入力を編集しても実行中の入力や結果ラベルは変わりません。完了数と経過時間は実際の応答から更新します。

prefix素材はmanifest hashを含むURLで配信し、ブラウザで1年間キャッシュ可能です。実行準備中に最初の2層を取得し、各層では次の2層まで先読みします。hash検証済みの32層（約15.4MB）はWorker内で再利用するため、同じページでの再実行時に取得・検証を繰り返しません。先読みは実行ごとにキャンセルでき、失敗した先読みは必要になった時に取得し直します。素材・manifestのhashやサイズが不一致ならHTTPキャッシュを迂回して1回だけ再取得し、それも不一致なら実行を中止します。破損した素材はWorker内にキャッシュしません。モデル、推論query数、ICへの中間状態とprefixの送信量は変わりません。

`Cancel` は通信を中断し、次のquery発行を止めます。すでにcanisterで実行されている計算の停止は保証しません。キャンセル後の古い応答は表示しません。queryは30秒、全推論は5分でtimeoutし、自動retryは行いません。失敗時はエラーを表示し、利用者が再実行できます。

Worker内で保持する素材の上限は15,418,368 bytesです。manifestとagentの準備、実行前の独立した3件の確認も並列化します。キャンセルは各実行の通信だけを中断します。仕組み、内部計測、比較結果は [FRONTEND_PREFETCH.md](../docs/FRONTEND_PREFETCH.md) を参照してください。

独立した入力・タブ・利用者の実行を直列化するqueueはありません。各実行がcarry・進捗・キャンセルを持ちます。モデル重み4.7GBをブラウザへ配布せず、ownerの秘密鍵やログインも必要ありません。IC agentのnode署名検証を有効にし、固定module/model/packとprefix素材のhashを確認します。host-checksum runtimeのzero footerは署名検証済み応答に限って受け入れ、全headerを送信条件に束縛した後、クライアントでchecksumを付けます。

BOOM #584・#653・#617は `data/boom-examples.json` からビルド時に短い英文を生成します。#584は「SNS Metadata Adjustment」というタイトルに対して、実際のactionは財庫から20M BOOMを指定口座へ送るTransferSnsTreasuryFundsです。「What would execution actually do?」にmetadata update / treasury transfer / no changeで答えます。提案はREJECTEDで、送金が実行されたとは表現しません。公開API原文を `data/boom-584-proposal.json` に保存し、ID・root・action・金額・財庫・口座・memo・status・原文一致を生成時に検証します。タイトルとpayloadの不一致を評価する例であり、提案者の悪意や攻撃者本人の同定は入力だけでは確定しません。

#653は実行済みのため、同額を現在追加発行する試算です。`npm run refresh:boom-ledger` で匿名の読み取りquery (`icrc1_total_supply` / `icrc1_balance_of` / `icrc1_decimals`) を送って、総供給量・対象のdefault subaccount残高・取得開始/終了時刻を `data/boom-653-ledger.json` に保存し、入力を再生成します。通常のbuildは保存値を使い、ネットワーク取得をしません。両値は別queryであり同一瞬間の状態ではありません。当時の提案判断を再現する値でもありません。ledger tokenは現在CHAIへ改名されていますが同じledger IDです。

モデル入力には総供給量をmillion単位、口座残高をtoken単位の包含区間にし、桁を削って96-token予算に収めます。同じ取得値から発行後の保有割合も機械計算し、包含区間を添付します。答えのyes/noは渡しません。正確な値と時刻、ledgerと対象口座は画面で確認できます。#617は「投票参加への主な影響」をrestricted access / wider access / no changeから判断する例です。`npm run refresh:boom-neurons` は公開indexed APIの全neuronをID順で取得します（limit=100、max_neuron_index固定、件数・重複・SNS照合）。保存先は `data/boom-617-neurons.json` で、build時に再集計し、原データのhashを保持します。現在のneuronに旧・新の最低ロック条件を適用し、positive indexed voting powerのneuron数、資格を失う件数、残る最大neuronの投票権比率を添付します。dissolving neuronは取得開始時の残存秒数で評価します。最大ロック期間も原文・auditで保持します。

neuron数は人数ではなく、現在のスナップショットによる試算です。ロックを延長せずに条件を適用した場合を評価し、提案当時の実際の影響や悪意は断定しません。集中度はAPI表示の現在の重みを固定した比較で、最大ロック期間変更によるbonusは再計算していません。ページ間の状態はatomicではありません。取得した5,699件では、資格を満たすneuronが2,219件→2件となり、2,217件（99.90%以上）が資格を失います。残る2件のうち1件のindexed投票権比率は99.99%以上です。`npm run test:boom617-live` は起動済み画面からこの入力の実推論を匿名queryで記録します。 2026-10-09のprefix27本番queryではrestricted accessを52.6%で選びました。入力・実応答・集計の出典は `artifacts/boom-617-participation-20261009/final/report.json` に保持しています。単一入力の動作確認であり、攻撃意図を判定するものではありません。原文と識別子の対応表も保持します。固定tokenizerとquery計画の上限でtoken数を検証し、上限超過や未対応の原文は生成エラーにします。自由入力と基本例は変換しません。`npm run test:examples` で数値範囲・保有割合の計算・原文・token数を確認できます。起動済みのローカル画面に対して `npm run test:boom-live` で#584と#653の実推論を匿名mainnet queryで検証できます。

2026-10-09のprefix27構成での本番canister検証では、#653はno 88.5%でした（以前の#660 Motion例はno 88.1%で、今回の攻撃関連候補へ差し替えました）。総供給量と残高だけの#653はunknown 66.9%だったため、ledger値から計算した発行後の保有割合も添付しています。算術前処理とモデルの判断を分けたデモで、モデル単体の計算能力を示すものではありません。入力と実応答は `artifacts/boom-enriched-demo-20261009/final/report.json` にあります。

入力例ボタンはフォームへ入力するだけです。`Copy input` は `state`、`question`、`options` のJSONをコピーします。履歴は永続保存しません。基本例のBF16ネイティブ実測は `artifacts/ui-extreme-examples-20261007/attempt2/` にあり、ICのINT8推論結果とは別の測定です。

## 検証

`npm test` は実tokenizerのPython出力との一致とPlaywrightのUIテストを実行します。

```sh
npm run test:query
PLAYGROUND_URL=http://127.0.0.1:4173 npm run test:live-query
```

`test:query` はGit管理外の5-tokenローカル検証証跡を使い、測定した実行計画のCandid送信byte、frame、中間返信、最終結果を照合します。複数回のMLP分割、破損・header不一致・キャンセルも検証します。移行と本番反映の順序は [PREFIX5_MIGRATION.md](../docs/PREFIX5_MIGRATION.md) を参照してください。

`test:live-query` は起動済みのサイトから本番へ読み取りqueryを送る明示的な試験です。独立した4ブラウザで短文・84token・85token・96tokenの異なる入力を同時実行し、update呼び出しがないこと、選択した計画の推論query数＋2readiness queryで完了することを確認します。84-token入力は既存の本番結果とも一致を確認します。2026-10-07のproduction build試験は両方成功し、87.346秒と67.498秒でした。証跡は `artifacts/browser-query-test-20261007/report.json` です。

全ての入力で命令数上限に収まることを保証する試験ではありません。instruction limit等で失敗した場合は結果を作らずエラーを表示します。低速回線・実スマートフォン・多数同時実行での性能は未測定です。

```sh
npx vlmkit check integrity http://127.0.0.1:5173
npx vlmkit check copy http://127.0.0.1:5173 --manifest verification/copy.txt
```

閉じたnative details内の検出は、展開時の可視性をPlaywrightで確認した上で当該selectorだけを除外します。

## 公開

公開queryのCandid定義と接続条件は `../docs/PUBLIC_QUERY_API.md` にまとめています。ビルド時に `/api/public-query.did` の配信素材を生成します。

公開先は https://imajev.kinic.xyz です。`cloudflare.config.ts` にCustom Domainを設定しています。Cloudflareの認証済みプロファイルとzone権限を確認して公開します。

```sh
npm run build
node scripts/prepare-cf-output.mjs
cf --profile kinic-production deploy --prebuilt
```

`npm run deploy` は同じビルド・成果物生成と、デフォルトプロファイルの `cf deploy --prebuilt` を行います。Custom Domainの設定だけでは公開先へ反映されません。`workersDev: true` により https://ic-imajev.hude.workers.dev からも同じfrontendを利用できます。

可変queryの測定、数値結果の比較、採用基準、32回未満の検討は [ADAPTIVE_QUERY_EXECUTION.md](../docs/ADAPTIVE_QUERY_EXECUTION.md) を参照してください。`scripts/profile-query-plans.mjs` は明示的な匿名mainnet query試験です。`scripts/prepare-query-profiles.mjs` は測定に合格した連続範囲だけを公開用の設定へ反映します。

2026-10-09の#584実推論ではtreasury transfer 62.1%を選び、metadata updateは24%、unknownは4.1%でした。86tokensで、本番canisterへの匿名queryだけを使い、update呼び出し・画面エラーはありませんでした。記録は `artifacts/boom-584-action-mismatch-20261009/report.json`。攻撃検出の一般精度ではなく、タイトルと実際の処理の不一致に対する1例の実測です。`BOOM_TEST_MODE=controls` を指定すると、同じタイトルで正当なロゴ変更と、タイトルをSNS Adjustmentへ変更した同じ送金命令の合成対照例を同じ選択肢・質問で検証できます。これらを実在の提案として表示しません。

合成対照例も同じ公開モデルで実測した。同じmetadataタイトルでロゴ変更だけのpayloadはmetadata update 93%、一般的なSNS Adjustmentタイトルで同じ20M送金payloadはtreasury transfer 85.6%。いずれも匿名queryのみで、update呼び出し・画面エラーなし。対照例の記録: artifacts/boom-584-action-mismatch-20261009/controls/report.json。3入力の結果は説明と実行内容を切り分けるこのデモの動作を示すが、一般的な攻撃検出精度の評価ではない。
