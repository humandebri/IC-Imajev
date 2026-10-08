# ブラウザのprefix取得と実行前確認

## 実装

Workerはprefix manifestのSHA-256を確認し、32層の素材を使います。実行準備中に最初の2層を取得し、層`k`を要求された時は次の2層まで先読みします。素材を現在のcanister query中に準備できます。ページを開いた時点ではprefix素材を取得しません。配信URLはmanifest hashを含み、immutable HTTP cacheを使います。

取得と検証が完了した`Uint8Array`だけをWorker内に保持します。キーは配信元URLと固定manifest hashで、保持するreleaseは1件、素材の合計上限は15,418,368 bytesです。ページを閉じると失われます。manifestに記載された32層、各素材の長さ・SHA-256を確認し、未検証の素材をキャッシュしません。queryの組み立ては共有素材を読み取り、書き換えません。

処理中のPromiseとAbortSignalは各実行に属します。別の実行へ処理中のPromiseを渡さないため、キャンセルした実行の通信を次の実行が待ち続けることはありません。同時実行では未取得の素材を重複取得する場合がありますが、共有キャッシュに保持するbufferは各層1個です。完了・エラー・キャンセルでその実行の先読み通信を中断します。先読みが通信エラーで失敗した場合は、必要になった時に取得し直します。hash・サイズが不一致の場合はHTTP cacheを迂回して1回だけ再取得し、それでも不一致ならエラーにします。ICへの推論queryは自動retryしません。

manifest取得とIC agent初期化を並列に実行し、入力を検証します。その後module証明、`pack_status`、`weight_cache_status`を並列に確認します。全て成功するまで推論queryを送りません。最終結果を表示する前のmodule証明確認も維持します。Candid、carry、token数ごとのquery計画、node署名検証、canisterのモデル・重み・moduleには変更がありません。

## 内部計測

`infer(input, progress, { onDiagnostics })`で任意のcallbackを指定すると、成功した実行について計測値が返ります。通常のUIでは指定せず、計測情報を表示しません。

- `prefix-fetch` / `prefix-hash`: manifestおよび素材の取得・検証。素材は`layer`付き。
- `prefix-wait`: query graphが素材を要求してから使えるまでの待ち時間。
- `prefix-cache`: 要求時点ですでに検証済みだった素材。
- `agent-init` / `module-check` / `query`: IC接続・証明確認・署名検証とCandid処理を含むquery時間。
- `preparation`: 実行前確認全体。`total`: Workerがrun要求を受けてから結果が用意できるまで。
- `prefixCacheBytes`: 実行成功時に保持している検証済み素材のbytes。

各フェーズは重なるため、合計して全体時間と比較しません。tokenizerが未準備なら`total`には初期化も含まれます。下記比較ではtoken数の確認でtokenizerを準備してから計測しています。

## 検証方法

`npm run test:query`は保存済みmainnet応答の再生で、元の送信byteと最終結果の一致、先読みの順序、warm時の取得・検証0回、破損素材の拒否、キャッシュ上限、キャンセルと再実行の独立性、並列確認の完了待ちを検証します。Playwrightでは既存の40件のUIテストを実行します。

`frontend/scripts/benchmark-prefetch.mjs`は明示的なmainnet読み取り試験です。計測を加えた変更前のproduction buildを4191、変更後を4192で配信し、短文と合計96-token入力についてcold/warmを各3回比較します。1組ごとに新しいブラウザcontextでHTTP cacheを消し、coldの後に同じWorkerでwarmを実行します。全24推論を直列に行い、組の順序を交互に入れ替えます。

この試験は保存済み結果との確率・選択値の完全一致、予定されたquery数、update呼び出し0件、warm素材の取得・検証0回を要求します。成果物には各WorkerのSHA-256、全フェーズと全結果を保存します。`BASELINE_URL`、`CANDIDATE_URL`、`PREFETCH_OUTPUT`で配信元と出力先を変更できます。baselineは同じdiagnosticsメッセージ形式を持つ計測用buildが必要です。

prefix静的配信はローカルVite preview、queryは実mainnetです。Cloudflareからの素材取得、低速回線、スマートフォン、多数同時実行の性能はこの比較では測りません。全体時間の中央値が入力・cold/warmのどの組でも5%を超えて悪化しないことを採用基準にし、少数回のネットワーク測定から一般的な高速化率を主張しません。

## 測定結果

以下はPR #11の1層先読み版の履歴です。現在ブランチでは2層先読み・immutable配信・素材再取得の実装に並列準備・計測・容量上限を統合しています。統合後の性能は再測定しておらず、この表をそのまま統合版の短縮率として扱いません。

2026-10-08に全24回が成功しました。全ての確率、選択値、判断保留状態が保存済み結果と一致しました。短文は34 query、96-tokenは65 query（各2 readinessを含む）で、update呼び出しは0件でした。

| 入力 | 実行 | 変更前の中央値 | 変更後の中央値 | 全体時間の差 |
| --- | --- | ---: | ---: | ---: |
| 短文 | 初回 | 51.14秒 | 49.15秒 | 3.9%短縮 |
| 短文 | 再実行 | 55.97秒 | 45.91秒 | 18.0%短縮 |
| 96-token | 初回 | 97.53秒 | 101.23秒 | 3.8%増 |
| 96-token | 再実行 | 117.63秒 | 103.91秒 | 11.7%短縮 |

全4組で中央値の悪化は5%以内でした。prefix待ち時間の中央値は短文初回179.8ms→12.2ms、96-token初回223.0ms→10.8msです。再実行は短文131.8ms→0.3ms、96-token131.8ms→0.2msでした。変更後の全初回で素材取得32回、全再実行で素材取得・hash検証0回、保持量15,418,368 bytesを確認しました。

素材待ちを減らす効果は確認できましたが、長い入力の初回は全体時間が増えています。通信やcanister負荷の変動を含む各3回の測定なので、再実行の全体時間の差も一般的な短縮率とは扱いません。

build hashと24回の計測・結果は [frontend-prefetch-20261008.json](benchmarks/frontend-prefetch-20261008.json)、全診断イベントはローカル証跡`artifacts/frontend-prefetch-20261008/report.json`に保存しています。変更前は`6fea57e`の逐次処理に任意計測だけを加えたbuildです。低速回線での素材配信は引き続き未測定です。
