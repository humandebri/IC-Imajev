# レビュー指摘3件の修正

2026-10-02。

- native bridgeに`module_hash`を追加。固定ic-agent 0.49.2の`read_state_canister_module_hash`で対象canisterの証明付きstateを読み、ローカルartifact・cacheのhashと照合する。cache load前、graph実行直前、graph完了後に照合。不一致はエラーで終了し、cacheは最終照合が成功してからsealする。
- 命令上限判定を`is_instruction_limit`へ共通化。IC0522と実際の「exceeded the limit of … instructions」を認識し、F32/整数/MLPのprojectionを半減して再試行する。モデル不一致等は再試行しない。既存のscheduler回帰テストを実エラー文へ変更した。
- head幅を要素数と2 MBのencoded blob予算から決める。DeltaNetのF32 state/gateを費用に含め、RoPE・attentionも分割する。prefix attentionはsuffix長ではなくKVを含む総token数で予算を決める。

Python30テスト成功。実canisterで147-tokenは7 heads、512-tokenは2 headsへ分割でき、単一headの同じcausal kernelと全bit一致。[境界比較](review-head-boundaries.json)。これはhead query単体の検証で、512-tokenの全モデル完走測定ではない。

新しい45-token cacheを実canisterから作り直し、617（132 tokens、処理suffix87）を611 queryで完走。全32層hidden・conv/DeltaNet/KV state・最終logit・確率・unknownが修正前と全bit一致。[全層比較](review-fixed-results.json)。Wasmは変更しておらず、実際のmodule hashも`806c1006b7effc726b0cb5ab77e7f4735fdbdb69308e6000d26856704e1f9664`と一致した。

module hash照合のread_state/証明取得は通常queryの推論費用とは別。計測JSONの通信はCandid request/replyであり、この証明通信を含めていない。実行中のcanister変更を検出するが、複数queryを一つの原子的なtransactionにはしない。

数値比較を行ったrunnerは`artifacts/review-fixed-runner-tested.py`にも保存した。その後、最終module照合をcache sealの直前へ移した（数値計算には変更なし）。生成物はGit対象外。

graph/source hashが変わったので、旧prefix cacheは再利用を拒否する。今回準備済みのcacheは`artifacts/review-fixed-prefix/queries`、実行結果は`artifacts/review-fixed-hit`。新環境では[DIRECTIONS.md](DIRECTIONS.md)の手順でcacheを再生成する。
