# 公開query APIの準備

本番canisterは `xis3j-paaaa-aaaai-axumq-cai`、networkは `ic`、gatewayは `https://icp-api.io`。公開アプリは https://imajev.kinic.xyz です。

`canisters/inference/public-query.did` はブラウザ推論に必要な6 queryの公開用subsetです。owner用update APIを含む全サービス定義とは区別します。既存の `paid-inference.did` は最新のAttention bridgeを含まないため、この経路の接続定義には使用しません。

| Query | 用途 |
| --- | --- |
| `pack_status` | model/pack identity、重みの準備状況 |
| `weight_cache_status` | immutable weight cacheの準備状況 |
| `step` | token IDsと固定prefixから最初のcarryを生成 |
| `mlp_delta_front` | streamed MLPと次のDelta層を実行 |
| `attention_mlp_front` | streamed MLPと次のAttention層を実行 |
| `terminal_step_decision` | 最終層とreadoutを実行して判断を返す |

本番は匿名queryを許可しています。`blob` は独自frame/carry形式であり、質問のUTF-8文字列をそのまま送る入口ではありません。入力は固定モデルのtokenizerでprompt全体をtokenizeし、共通prefix27＋追加1〜57 token、合計28〜84 tokenに制限します。選択肢は2〜7件、重複・空欄・予約語 `__unknown__` を拒否し、各128 UTF-8 bytes以内です。

一つの推論は32回を依存順に呼びます。異なる推論のcarryは別々に保持してください。prefix素材は `frontend/public/inference/prefix27-v1/` に生成されます。model/pack/module/manifestの固定値は `frontend/src/inference-release.json` にあります。Candidだけでは数値計算の手順や応答の検証を実装できないので、既存 `frontend/src/query-runner.ts`、`query-codec.ts`、`inference-agent.ts` を合わせて参照してください。

`frontend/src/inference-client.ts` の `infer({ state, question, options }, onProgress)` がWeb Worker経由の入口で、返り値の `promise` と `cancel()` を使います。Vite等のWorker bundlingとtokenizer/prefix静的素材の配信が必要です。公開済みnpm SDKではありません。進捗と結果だけをUIへ渡し、node署名の検証を無効にしたり、zero footerの応答を未検証のまま受け入れたりしないでください。

## 定義の生成と検証

```sh
cd frontend
node --experimental-strip-types scripts/prepare-public-candid.mjs --check
didc check ../canisters/inference/public-query.did
```

agentが使うCandid型から生成し、定義がずれるとdev/build/testの前処理で失敗します。型を変更する場合だけ `--write` でcanonical定義を更新し、差分をレビューしてください。

```sh
icp canister call xis3j-paaaa-aaaai-axumq-cai pack_status '()' \
  --query --network ic --identity anonymous \
  --candid canisters/inference/public-query.did
```

build時に公開用ファイルを `/api/public-query.did` にコピーします。frontendを次回deployすると https://imajev.kinic.xyz/api/public-query.did で配信できます。今回の作業は公開準備であり、このURLへの反映はまだ行っていません。

## canister自身からの自動取得

現在、本番moduleには `candid:service` metadataがありません。CLIやCandid UIに自動検出させるには、最新の全サービス定義を生成してpublic metadataに埋め込み、runtimeをupgradeする必要があります。subsetを全サービス定義として埋め込まないでください。

metadataを追加するとmodule hashも変わります。upgrade後はheapに置いたweight/prefix/query cacheの再準備と、frontendのmodule hash更新・query再検証が必要です。公開Candidの配信だけならcanisterのupgradeは不要です。

## 公開前後の追加作業

- cycles残高の監視と補充通知。2026-10-08の補充直後は凍結閾値まで約56日ですが、固定日数だけに頼らずlive balanceと日次保管費を監視する。
- 「raw Candid」と「テキスト入力から推論を実行するクライアント」の利用例をセットで案内する。32回は検証済みの経路であり、全入力で命令数上限に収まる保証はしない。
- module hashと静的prefix素材を一緒に管理し、upgrade時の再準備・frontend更新・戻し方を運用手順にする。
- 公開サイトの疎通と、実スマートフォン・低速回線での完了時間やキャンセルを確認する。全利用者のqueueや並列数制限は今回の設計には追加しない。

自動監視・追加upgrade・frontend公開は、この準備では実行していません。
