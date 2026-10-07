# IC-Imajev 判断プレイグラウンド

ブラウザから公開canister `xis3j-paaaa-aaaai-axumq-cai` に匿名queryを送り、実推論の結果を表示します。tokenizerとqueryの組み立ては同じWeb Workerで実行します。Cloudflareは静的ファイルを配信します。

## 起動とビルド

Node.js 24以上、npm、リポジトリの `.venv/bin/python` とNumPyが必要です。`MODEL_LOCK.json` のtokenizer・readoutファイル、`checkpoints/full-int8.manifest.json`、検証済みprefix archive `artifacts/query-packing-v3/prefix-v2/` を準備してください。モデル重み本体はfrontendのビルドに不要です。

```sh
cd frontend
npm ci
npm run dev
```

http://127.0.0.1:5173 を開きます。`dev`、`build`、`test` の前処理で固定tokenizerのhashとprefix archiveのidentity・各ファイルのhashを検証し、静的素材を生成します。tokenizerは約12.8MB、prefix素材は15,418,368 bytesです。`public/tokenizer/` と `public/inference/` はGit対象外で、ビルド成果物に含まれます。

```sh
npm run typecheck
npm run build
npm run preview
npm test
```

PlaywrightのChromiumがない場合は `npx playwright install chromium` が必要です。

## 入力と実行

質問は必須、判断材料は任意、選択肢は2〜7件です。空欄、前後空白を除いた重複、1選択肢128 UTF-8 bytes超は拒否します。固定27-token prefixと追加1〜57 token、合計28〜84 tokenが対象です。prompt、選択肢、unknown、chat templateを含む実token数で実行可否を判断します。

`Run inference` で送信時の入力を固定し、32回の推論queryをcarryの依存順に呼びます。実行前に2回のreadiness query、実行前後にmodule hashの証明を確認します。結果には選択値、候補別確率、unknown、判断保留を表示します。途中で入力を編集しても実行中の入力や結果ラベルは変わりません。完了数と経過時間は実際の応答から更新します。

`Cancel` は通信を中断し、次のquery発行を止めます。すでにcanisterで実行されている計算の停止は保証しません。キャンセル後の古い応答は表示しません。queryは30秒、全推論は5分でtimeoutし、自動retryは行いません。失敗時はエラーを表示し、利用者が再実行できます。

独立した入力・タブ・利用者の実行を直列化するqueueはありません。各実行がcarry・進捗・キャンセルを持ちます。モデル重み4.7GBをブラウザへ配布せず、ownerの秘密鍵やログインも必要ありません。IC agentのnode署名検証を有効にし、固定module/model/packとprefix素材のhashを確認します。host-checksum runtimeのzero footerは署名検証済み応答に限って受け入れ、全headerを送信条件に束縛した後、クライアントでchecksumを付けます。

入力例ボタンはフォームへ入力するだけです。`Copy input` は `state`、`question`、`options` のJSONをコピーします。履歴は永続保存しません。基本例のBF16ネイティブ実測は `artifacts/ui-extreme-examples-20261007/attempt2/` にあり、ICのINT8推論結果とは別の測定です。

## 検証

`npm test` は実tokenizerのPython出力との一致とPlaywrightのUIテストを実行します。

```sh
npm run test:query
PLAYGROUND_URL=http://127.0.0.1:4173 npm run test:live-query
```

`test:query` はローカルの本番証跡 `artifacts/mainnet-prefix27-upgrade-20261007/anonymous-query-653/` と `artifacts/text-short-v2/inputs.json` を使います。32回のCandid送信byte、frame、中間返信、最終結果を照合し、破損・header不一致・キャンセルを検証します。

`test:live-query` は起動済みのサイトから本番へ読み取りqueryを送る明示的な試験です。独立した2ブラウザで異なる入力を同時実行し、update呼び出しがないこと、各32推論query＋2readiness queryで完了することを確認します。84-token入力は既存の本番結果とも一致を確認します。2026-10-07のproduction build試験は両方成功し、87.346秒と67.498秒でした。証跡は `artifacts/browser-query-test-20261007/report.json` です。

全ての入力で命令数上限に収まることを保証する試験ではありません。instruction limit等で失敗した場合は結果を作らずエラーを表示します。低速回線・実スマートフォン・多数同時実行での性能は未測定です。

```sh
npx vlmkit check integrity http://127.0.0.1:5173
npx vlmkit check copy http://127.0.0.1:5173 --manifest verification/copy.txt
```

閉じたnative details内の検出は、展開時の可視性をPlaywrightで確認した上で当該selectorだけを除外します。

## 公開

`cloudflare.config.ts` の希望公開先は https://imajev.kinin.xyz です。Cloudflareの認証済みプロファイルとzone権限を確認して公開します。

```sh
npm run build
node scripts/prepare-cf-output.mjs
cf --profile kinic-production deploy --prebuilt
```

`npm run deploy` は同じビルド・成果物生成と、デフォルトプロファイルの `cf deploy --prebuilt` を行います。Custom Domainの設定だけでは公開先へ反映されません。今回の実装・テストではfrontendの公開操作は行っていません。
