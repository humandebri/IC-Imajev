# IC-Imajev 判断プレイグラウンド

API接続前の入力フォームと結果表示領域です。実推論、モック判定、架空の結果、課金処理はありません。「判定する」は無効です。

## 公開ドメイン

希望する公開先は https://imajev.kinin.xyz です。`cloudflare.config.ts` の `worker.domains` に設定しています。現在の `kinic-production` 認証では `kinic.xyz` のzoneは確認できましたが、`kinin.xyz` は見つからず、カスタムドメインの適用は未完了です。

```sh
cd frontend
npm run deploy
```

`npm run deploy` はフロントエンドをビルドし、`scripts/prepare-cf-output.mjs` で静的配信用成果物を生成してから、`cf deploy --prebuilt` で公開します。名前付き認証を使う場合は、`npm run build` → `node scripts/prepare-cf-output.mjs` → `npx cf --profile kinic-production deploy --prebuilt` の順に実行します。

デプロイには `kinin.xyz` のCloudflare zoneへの権限が必要です。Custom DomainのDNSレコードとTLS証明書はCloudflareが管理します。設定の追加だけでは公開先に反映されません。

## 起動

Node.js 24以上とnpmを使います。

```sh
cd frontend
npm ci
npm run dev
```

ブラウザで http://127.0.0.1:5173 を開きます。

```sh
npm run typecheck
npm run build
npm run preview
npm test
```

PlaywrightのChromiumがない場合は `npx playwright install chromium` が必要です。

## 入力と結果

判断材料は任意、質問は必須、選択肢は2〜7件です。空欄または前後空白を除いた重複はエラーとなり、入力コピーを無効にします。コピーするJSONは `state`、`question`、`options` のUI入力形式で、token化されたCandidリクエストではありません。入力はページ内だけに保持され、再読み込みで初期化されます。

基本の3例は、月額料金の $10 → $10,000 への変更、配達予定日と現在地の不明な荷物、製品は最高・配達は最悪というレビューです。タブは `Value change` / `Missing info` / `Three choices`。正解・モデル出力をフォームに埋め込みません。実際のBOOM DAO proposalのpayload原文抜粋は、折りたたみの `Real-world examples` に分け、出典リンクとデモ用の質問である旨を表示します。

この3入力を、固定された公式ネイティブMLXのQwen3.5-4B + adapter、同じ `text-only-short-v2`、元の温度校正で測定しました。結果は `yes`（97.68%）、`unknown` による判断保留（99.32%）、`mixed`（86.07%）。これはネイティブBF16/F32の実測で、ICのINT8版の結果ではありません。ローカルICはstatus取得がタイムアウトしました。確率は校正済み候補スコアであり、この3例だけから一般的な正確さは主張しません。[完全精度の実測とlogits](<../artifacts/ui-extreme-examples-20261007/attempt2/results.json>)、[モデル・実行環境の証跡](<../artifacts/ui-extreme-examples-20261007/attempt2/provenance.json>)。公開UIのAPI接続状況は変えていません。

通信量の注意書きは画面に表示しません。約71〜79 MBは `docs/QUERY32_PROGRESS.md` の既存短文3入力（固定prefix準備済み）のローカル測定に基づくCandid送受信合計です。HTTP等の追加通信、初回準備、再試行を含む上限値ではありません。query接続はまだ実装していません。

`ResultPanel` は実応答の選択値、候補別確率、unknown、判断保留を表示できるコンポーネントです。現在は `null` のみを渡して未接続表示にしています。接続時は応答と同じリクエストの選択肢を渡してください。tokenizer、prompt生成、token制限、中継canisterは未実装です。

実行中は `ResultPanel` の `progress` に実際の進捗だけを渡します。固定重み・prefixの準備は公開前に済ませるため数えません。通常queryを順に呼ぶ推論経路は `{ kind: "steps", completed, total, startedAt, phase? }` で「完了数 / 予定数」・経過・残り目安（同じ実行の完了済みqueryの平均から算出、3回完了まで「計測中」）を表示します。有料updateのように1回の呼び出しで内部の進捗が見えない経路は `{ kind: "waiting", startedAt }` で経過時間だけを表示し、進捗率は出しません。結果を受け取ったら `progress` を `null` にして `result` を渡します。

## UI検証

```sh
npx vlmkit check integrity http://127.0.0.1:5173
npx vlmkit check copy http://127.0.0.1:5173 --manifest verification/copy.txt
npx vlmkit scan scroll http://127.0.0.1:5173
npx vlmkit scan handlers http://127.0.0.1:5173
npx vlmkit check interactions http://127.0.0.1:5173
npx vlmkit check breakpoints http://127.0.0.1:5173 --sweep
```

## トークン数

固定モデルの `tokenizer.json` / `tokenizer_config.json` とreadoutコードを使い、ブラウザのWorker内で数えます。`text-only-short-v2` のprompt、選択肢、unknown、chat templateを含んだ合計・共通prefix・追加tokenを表示します。有料updateの57-token上限はqueryの上限とは分けて表示します。入力を外部サービスへ送る処理はありません。

`npm run dev` / `npm run build` / `npm test` の前に、`MODEL_LOCK.json` のhashを検証して `checkpoints/` からtokenizerをコピーします。これらの固定ファイルがないcheckoutでは事前にモデルのtokenizerとreadoutファイルの準備が必要です。生成先 `public/tokenizer/` はGit対象外ですが、ビルド成果物に含まれます。モデル重みは不要です。tokenizerの初回読み込みは約12.8 MBです。

`npm run test:tokenizer` でPythonの既存 `TextPreparer` と全token ID列を照合できます。`tests/tokenizer-parity.json` は既存入力例、日本語、空の判断材料、引用符・改行の実tokenizer出力を保存した検証用データです。推論のモックではありません。

画面は入力を左、実応答の表示領域とtoken数を右に配置します。初期表示では重複する説明や回答候補プレビューを省き、token数は合計だけ表示します。内訳と有料APIの条件は「カウントの詳細」で確認できます。入力例はフォームへの反映だけを行います。
