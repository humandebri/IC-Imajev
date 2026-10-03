# 活性化関数の固定準備

2026-10-03。毎要素のexp・除算・logを固定準備へ移す実装を全5条件で検証し採用。主問題のhandler命令を7,547,648,244（2.6956%）削減。主67 queryと通信量は同じ。50/32 queryは未達。

## 採用済みの全モデル実測

前回[Delta full log版](DELTA_FULL_LOG.md)との比較。全5条件の保持hidden・状態配列・判断・確率がbit一致。prefixは元F32状態へ可逆復元した比較に加え、保持log自体のbitも一致。失敗/replay0。実行前後の認証module hash・固定cacheと50 core source hashは一致し、保存source archiveとも一致した。

| 条件 | 通常query数 | handler命令: 元→固定表 | 削減率 | Candid通信byte | 単回秒: 元→固定表 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix45 token | 66 | 151,805,703,029→147,852,713,772 | 2.6040% | 71,105,691 | 17.913→17.359 |
| 主 suffix87 / 全132 token | 67 | 279,999,674,508→272,452,026,264 | 2.6956% | 113,709,005 | 31.929→30.549 |
| 情報不足 suffix80 / 全125 token | 67 | 259,814,890,111→252,882,014,904 | 2.6684% | 106,668,202 | 29.222→28.559 |
| 重大変更 suffix89 / 全134 token | 98 | 306,318,620,639→298,608,130,562 | 2.5171% | 197,070,502 | 35.243→33.959 |
| prefixなし132 token | 292 | 447,558,340,370→436,125,574,914 | 2.5545% | 580,229,781 | 57.558→55.562 |

初回prefix+主問題は133 query、420,304,740,036命令（11,500,637,501命令、2.6634%減）、184,814,696 Candid byte、今回単回47.907秒。通信量は前回と同じ。HTTP/CBOR/署名を含めない。handler counterはCDK Candid decode/encodeを含めない。単回時間の比較は負荷を統制した反復実験ではなく、一般的な速度向上を保証しない。

主問題の最大queryは4,869,416,555→4,723,869,983命令。全5条件の最大終端heap観測は4,123,197,440 byte、以前より1,114,112 byte増。4 GiB設定以内で完走したが、一時peakを保証する値ではない。固定cacheは重み4,065,416,192 byte、RoPE131,072 byteに活性化表1,048,576 byteが加わる。

今回の固定準備は721 owner update、15,077,999,700命令、256.111秒、request47,473/reply14,938,321 Candid byte。初回updateで表も準備する。pack status1 query、cache status2 query、認証module read2回は別記録。全5条件を挟むcache確認2 queryと認証module read2回も推論query数へ含めない。

採用Wasm SHA256: `ba47cf87e84e980c6b4d20fab6e7aec88fee110ee5d9b2edfa92dcaa3ba15e7e`。canister `4caro-hl777-77775-aaaba-cai`。pack SHA256 `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`、モデル組・token数・INT8 block256・元F32 adapter/readout・calibrationは維持した。

証拠は`artifacts/prepared-activation-v1-cache-checks/report.json`、各`artifacts/prepared-activation-v1-*/report.json`、比較`docs/prepared-activation-v1-*-results.json`、集計`docs/prepared-activation-v1-summary.json`。`artifacts/prepared-activation/validated-source.zip`と`source-hashes.json`で50 core sourceを保存。prefix状態のオフライン復元は検証専用で、運用クライアントの推論へ結果を渡さない。

既存の重大変更gold=yes/model=noと元BF16公式実装との差は残る。型安全な出力と判断精度は別に評価する。今回のbit一致は以前からの追加劣化が検出されなかったことを示し、一般的な精度向上を示さない。

## 残るボトルネック

主問題のMLP31 queryは150,307,111,120→146,407,420,199命令、Delta24 queryは97,439,705,183→94,207,787,615命令、Attention8 queryは31,887,960,801→31,473,546,958命令。ほかembed・norm・終端MLP・readoutが4 query。MLPがhandler命令の約53.74%を占める。固定表は非線形演算を減らすが、整数dotの積和そのものは維持する。

主問題のhandler合計272.452Bだけで5B×50を超えるため、分割境界の変更だけで50 queryへ到達したとは言えない。50にはhandler合計をさらに少なくとも約8.24%、32には約41.27%減らす必要があり、CDK decode/encodeの余裕と分割状態も別途必要となる。この値は現在実装の集計からの必要条件で、演算削減方式の理論限界ではない。

## 固定表の仕様

BF16の有限入力65,280通りについて、元の`bf_sigmoid`、`bf_silu`、`bf_softplus`、F32版`silu`の出力bitを固定表へ保存する。Wasm自身が元の演算で表を生成するため、native/Wasmの既存の数学関数差を持ち込まない。入力のBF16丸めを追加する処理はない。非BF16、非有限値、表未準備の場合は元の演算へfallbackする。F32版SiLUは元の`x * sigmoid(x)`の結果を保存し、除算形へ式変形しない。

固定表は1,048,576 byte。準備はowner update、推論は通常queryで参照するだけ。質問に依存した中間状態は保持しない。`warm_weights`で固定表も一度準備し、`weight_cache_status.activation_bytes`で準備完了を確認する。`clear_weight_cache`は固定表も破棄する。upgradeではheap表が失われるため再準備する。

## 独立したローカルcanisterでの検証

canister `7vs54-wt777-77775-aaajq-cai`。24通常queryで3入力集合×4関数×元/lookupを比較し、各集合の全出力SHA256が一致。全有限BF16、合成範囲−8〜8の131,072値、非BF16・非有限値8値を検証した。合成範囲は実際のモデル内分布を採取したものではない。

| 関数 | 全有限BF16: 元→lookup kernel命令 | 合成−8〜8: 元→lookup kernel命令 |
| --- | ---: | ---: |
| bf_sigmoid | 9,339,805→4,469,461 | 31,322,827→8,871,733 |
| bf_silu | 10,841,245→4,489,461 | 34,337,483→8,891,733 |
| bf_softplus | 17,009,399→4,489,461 | 47,803,115→8,891,733 |
| silu | 7,072,981→4,534,741 | 25,817,803→9,022,805 |

fallbackでは入力判定の分だけkernel命令が増加する（8値で92〜148命令）。元の計算の出力bitは維持する。表準備のupdateは34,458,960 handler命令。query kernel counterにはCandid decode、入力byte復元、出力SHA256、reply encodeを含めない。kernel以外も記録したhandler counterにもSHA256とreply encodeは含めない。単回時間から一般的な速度差を推定しない。

Rust nativeでも全有限BF16×4関数とfallback入力をbit比較。推論featureのRust検証はunit64、INT8 integration3、F32 integration2、compile-fail3が通過した。Rust検証と同Wasm比較を分け、backendをまたいだbit一致は要求しない。

診断sourceは`artifacts/prepared-activation/validated-diagnostic-source.zip`、proofは`artifacts/prepared-activation/check/report.json`。実行前後の認証module hashとsource hashを照合した。生成物はgitignore対象。

## 再現手順

推論用Wasmの既存feature集合に`experimental-prepared-activation`を加える。`prepare_weight_cache.py`と`validate_prepared_weights.py`に`--require-prepared-activation`を加え、表が1,048,576 byte準備されていることを検証する。未準備fallbackを使った推論を採用結果として受け入れない。

診断用ビルドは`cargo build --offline --release --manifest-path scripts/activation_bench/Cargo.toml --target wasm32-unknown-unknown --target-dir artifacts/prepared-activation/wasm --lib`。Candid補助CLIは同manifestの`activation_args`をnativeにビルドする。owner付きで新しいローカル診断canisterへinstallし、`scripts/check_prepared_activation.py --canister <id>`を実行する。
