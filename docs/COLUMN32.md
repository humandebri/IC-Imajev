2026-10-03後続：全32 headのDelta統合を全5条件で検証・採用し、主91→67 query。現在の採用版は[DELTA_FULL_LOG.md](DELTA_FULL_LOG.md)。以下は探索・採用時の履歴。

# 入力ロードの32出力共有と87トークンMLPの統合

2026-10-03。毎回繰り返す入力ロードと、分割境界での中間状態の返信・復元を減らす実装。INT8 base block256、元F32 adapter/readout、既存のF32積和順序を維持する。

## 変更

`int8_column32.rs`は16出力用kernelの共有範囲を32出力へ広げる。各256要素blockの入力SIMDロードを32出力で使い回す。重みを8要素ずつ符号拡張し、I32内積の後に従来通り`(dot as f32 * sx) * sw + sum`を行う。入力・重みの量子化方式、浮動小数点の加算順、LoRAは変更しない。Laya由来部分のMIT帰属はファイル冒頭と`docs/licenses/Laya-MIT.txt`に保持する。

対象は入力が81〜88 token、出力行数が32の倍数の場合。内部では44+44 tokenのtileを用い、それ以外は既存のbalanced44/16出力経路を維持する。87 tokenの末尾padding 1行は引き続き計算しており、そこは未削減。

実測で87 tokenの全MLPが通常queryの上限へ収まったため、`mlp_full_integer`の入力上限を80から87へ広げた。clientも同じ上限を用いる。gate/up/down・残差・次層normを1 queryへつなぎ、中間状態の返信・再送・frame復元を除く。最終層の専用readout経路は従来通り。

固定モデル重みとRoPE表の準備にはupdateを使う。質問に依存する中間状態はクライアントが保持し、推論は通常queryで進める。

## 採用済みの全5条件実測

旧採用ByteBuf v3との比較。全5条件の保持hidden、prefix 72/各質問48の状態配列、質問の判断・確率がビット一致。失敗/replayは0。実行前後の認証module hash、固定cache、44 source hashが一致し、source archiveとの一致も確認した。

| 条件 | query | handler命令数 | Candid通信bytes | 単回秒 | 最大handler命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix 45 token | 90→90 | 156,313,221,966→156,313,227,846 | 145,902,403→145,902,403 | 20.748→20.633 | 2,697,997,814 |
| 主問題 suffix87 / 全132 token | 122→91 | 287,186,913,161→279,591,007,008 | 262,955,428→183,432,765 | 35.148→32.945 | 4,869,416,555 |
| 情報不足 suffix80 / 全125 token | 91→91 | 259,182,862,243→259,182,867,349 | 173,060,522→173,060,522 | 30.851→31.508 | 4,520,882,576 |
| 重大変更 suffix89 / 全134 token | 122→122 | 305,987,635,365→305,987,641,245 | 267,767,242→267,767,242 | 37.079→37.812 | 4,767,482,915 |
| prefixなし 132 token | 292→292 | 447,558,326,622→447,558,332,080 | 580,229,781→580,229,781 | 78.287→58.083 | 2,882,723,905 |

主問題は31 query（25.410%）、7,595,906,153命令（2.6449%）、79,522,663 byte（30.2419%）削減。prefix込み初回は212→181 query。prefixなし132 tokenは292 queryのまま。対象外条件には合計5,106〜5,880命令の分岐コスト増がある。単回時間はキャッシュや負荷に左右され、対象外条件にも大きい時間差があるため、速度改善の保証とは扱わない。

主問題の最大queryは4,869,416,555命令で実queryは完走した。handler counterにはCDKのCandid decode/encodeを含めない。最大終端heap観測は4,119,134,208 byte、固定weight cacheは4,065,416,192 byteとRoPE 131,072 byte。終端heap観測は実際の一時peakを保証しない。通信量はrequest/replyのCandidを合算し、HTTP/CBOR/署名は含めない。

固定準備はモデル/upgrade時のみ。今回721 update、15,041,482,761命令、239.857秒、request47,473/reply14,927,506 byte。別にpack status 1 query、cache status 2 query、認証module read 2回。全5検証のcache確認2 query・認証module read2回も推論queryとは分けて記録している。

採用Wasm SHA256: `e3d8b36acdd4a715008d8ea7a6a8d3d9d67720fcb45c9df0c4c049b7b2587c9c`。pack SHA256: `364e3aa3f69a61a69e6df62e3e2a71d4f55457f1a8e1c16934aefc6130f04c84`。

全モデルproof: `artifacts/column32-v1-cache-checks/report.json`、各実行`artifacts/column32-v1-*/report.json`、比較`docs/column32-v1-*-results.json`、集計`docs/column32-v1-summary.json`。候補Wasm/source archiveは`artifacts/column32/`。これら生成物はignore対象。

50/32 queryは未達。元BF16公式実装との既存差や重大変更ケースのgold=yes/model=noの誤判定は残る。型安全な出力が有効であることと、判断の正しさは別に評価する。今回のビット一致は最適化前からの劣化を検出しないことを示し、一般的な精度改善を示さない。

## 残るボトルネック

主問題のhandler命令ではMLPが150,307,111,479（53.76%）、Delta前後が97,031,037,257（34.70%）、Attentionが31,887,960,868（11.41%）。主91 queryのうちDeltaが48、MLPが31、Attentionが8、embed/norm/終端readout等が4。

毎回の小さいmetadata処理だけでは、MLP/Deltaの積和を大幅には減らせない。次は出力共有範囲の拡大、padding 1行の不要計算、固定重みのロード共有を実queryで比較する。Delta全32 headのF32 stateだけで2,097,152 byteに達するため、現在の入力/返信をそのまま1 queryへ統合できない。client-held状態の可逆表現を変更する案は、通信上限と復元命令の両方を測定してから判断する。

## 段階検証

- 独立scalar・従来16出力kernelとのnative比較、signed INT8端値、token端数、異常scale/weight長を検証。runtime 60 tests、INT8 integration 3、F32 integration 2、compile-fail 3、client MLP routing 4が通過。
- 専用診断canisterで29条件・58通常query。5つの実入力と24境界条件で、候補・従来Wasm・独立nativeがビット一致。主87 tokenのQ base投影は1,119,201,387→1,089,231,254命令（2.6778%減）。対象外の実形状45/80/89/132 tokenは分岐により45命令増。これはbase投影単体の測定で、推論全体の削減率ではない。
- 主87 tokenの全31層、およびprefix/情報不足の境界層を計36通常queryで比較。全出力が以前のWasmの2 query MLP出力とビット一致。最大handler命令数4,869,416,555。不正token数・layer・epsilon・next normの4 queryを拒否。

段階検証のraw reportは`artifacts/column32/check/report.json`と`artifacts/column32/partial/report.json`。診断ソースは12 hash、全モデルの候補ソースは44 hashでarchiveに固定する。生成物はGit管理しない。

## 他の方向の探索

固定LoRA A/B 400 tensorを走査したが、全ゼロtensorと同一shape・同一内容の重複はどちらも0だった。adapter全体の省略や重複共有による削減は適用できない。

Relaxed SIMDの整数dotを使う候補については、精度を保つ整数分解を確認し、151 byteの最小Wasmを新しい専用canisterへinstallした。現行ローカルICが「relaxed SIMD support is not enabled」で拒否したためモデルkernelには採用しない。実probeと公式実装へのリンクは[RELAXED_SIMD.md](RELAXED_SIMD.md)に記録した。

## 再現

準備済み専用canisterに対する全5条件の比較:

```sh
.venv/bin/python scripts/validate_prepared_weights.py \
  --canister 4caro-hl777-77775-aaaba-cai \
  --run-name column32-v1 --baseline byte-buffer-v3 \
  --wasm artifacts/column32/full.wasm \
  --preparation artifacts/column32/preparation \
  --reuse-projection-inputs --frame-checksum blake3 \
  --fuse-attention --fuse-attention-full \
  --fuse-delta-projected --fuse-delta-finish \
  --fuse-mlp-pipeline --fuse-mlp-full --require-prepared-rope
```

buildは従来の全最適化featureに`experimental-column32`を追加する。`experimental-mlp-full`もruntimeのcolumn32採用とcanisterのbyte bufferを自動で有効にする。旧モデルpackは変更しない。
