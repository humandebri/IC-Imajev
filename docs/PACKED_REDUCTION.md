# 整数結果のSIMD取り出しとDelta係数共有の拡大

2026-10-02。前版 `delta-lanes-v1` から、二つの出力保存型の改善を採用した。

* 整数射影の16/32/64-token tileで、四つのdot結果の水平和をまとめてSIMD vectorに格納する。各結果をスカラーへ取り出す代わりに、shuffle/addで四つの合計を作る。I32積和の範囲は従来と同じで、block256 scaleのF32演算順とBF16境界は変更しない。8 token以下の既存経路は保存する。
* Delta再帰のvalue幅128では、32個のSIMD vectorでK/Q係数を共有する。幅128未満には前版の16要素処理を残す。key軸の積和順、decay、F32状態は変更しない。

実Q投影8192×2560の6形状で、以前の出力とnative scalarに全bit一致した。1/7/8 tokenの命令数は同じ、32 tokenは3.17%、64 tokenは1.63%、87 tokenは1.85%減った。別のDelta synthetic 12条件も出力・状態が両参照に全bit一致し、87 token・128幅では52,611,247→46,051,278命令（12.47%減）。小幅では以前の経路を使い、当初候補の幅16で約18%増える問題を回避した。単体削減率は全モデルの削減率と区別する。

固定4B重みを保持した主canister `4caro-hl777-77775-aaaba-cai` で、新prefixから全5実行を再検証した。保持hidden/state、最終hidden/logits/確率/unknown/判断は前版と全bit一致。prefixは72状態配列、各問題は48状態配列を比較し、最終層で省略する位置のhiddenを一致対象に含めていない。失敗・replayは0、全実行で前後の認証済みmodule hashが一致した。

| 実行 | query | 前版の命令数 | 改善後の命令数 | 削減 | 通信bytes | 時間s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix45 | 330 | 221,068,499,713 | 213,588,222,593 | 3.38% | 360,695,462 | 43.730 |
| 主問題132 / suffix87 | 352 | 389,142,274,450 | 379,487,952,658 | 2.48% | 579,259,841 | 52.978 |
| 情報不足125 / suffix80 | 352 | 343,284,477,177 | 334,035,516,537 | 2.69% | 537,098,941 | 51.036 |
| 最大変更134 / suffix89 | 352 | 401,899,603,710 | 390,668,578,046 | 2.79% | 591,305,752 | 78.345 |
| cacheなし132 | 501 | 568,743,949,014 | 554,462,815,702 | 2.51% | 885,497,502 | 86.716 |

query数・通信量は同じ。初回主問題はprefix準備330＋推論352＝682 query。主問題の最大query命令数4,104,518,761、観測heap43,450,368 bytes。時間は単発実測で、前版より遅い実行もあるため速度改善とは報告しない。handler命令数はCandid decode/encodeを含まず、通信量はCandid request＋replyでHTTP等を含まない。

50/32 queryは未達。主問題の命令数だけを各queryの5B予算で割っても最低76 queryに相当する。50 queryに向けてさらに約34.1%、32 queryには約57.8%の命令削減に加え、実際のquery統合とメッセージサイズ制約の解決が必要。最大lock問題の既存誤判定は残る。判断精度や校正が改善したという結果ではない。

生成Wasmも静的に比較した。64×8整数tileではdot数16,384は同じ、shuffleは1,280→768、SIMD storeは256→128。ただしlocal getは35,190→35,864、local数も361→2,316に増えた。組み替えの削減だけから総命令数や実時間を推定せず、canister実測を採用判断に使った。ICの[公式metering実装](https://github.com/dfinity/ic/blob/master/rs/embedders/src/wasm_utils/instrumentation.rs)でもlocal get/set/teeは費用を持つが、masterの静的費用をこのreplicaのバージョン一致の証明には使わない。

既定Wasm SHA256: `7c366d0b7958a3034b89dff98c0fe1486239d3419cad4882324231c7268b59b5`。Rust33／Python40 tests通過。実重み・非ゼロ出力row offset・token境界・元F32 LoRAを含む126条件でnative scalarとWasmを追加照合し、全bit一致した。生成物はgitignore対象、Laya・mainnet・Git remoteは変更していない。

生測定 `artifacts/packed-reduction-v1-*`、集計 `docs/packed-reduction-v1-summary.json`、整数単体 `artifacts/packed-reduction/{before,final-probe}`、Delta単体 `artifacts/delta-lanes32/{before,final}`、126条件 `docs/packed-reduction-integer-parity.json`。候補生成は `scripts/generate_packed_reduction.py` と `scripts/generate_delta_group.py`、単体実測は `scripts/probe_partial_projection.py` と `scripts/check_delta_lanes.py`。

再現時は未使用run名で `scripts/validate_terminal_readout.py --run-name <新しいrun名> --baseline delta-lanes-v1 --canister <専用local ID>` と `scripts/summarize_kernel_unroll.py --run-name <同じrun名> --baseline delta-lanes-v1` を実行する。初回prefix準備の費用を省略しない。
