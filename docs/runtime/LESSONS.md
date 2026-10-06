# 他モデルへ持ち出す効率化の知見

再利用の中心は、同じ仕事を繰り返さず、数値契約を守ったままメモリと実行境界を変えることにある。削減率は対象形状・backend・直前版に依存する。

| 手法 | 移植時に保存する契約 | 根拠 |
|---|---|---|
| 入力量子化・LoRA A積の共有 | block境界、scale、元F32入力 | [MLP_PIPELINE](../MLP_PIPELINE.md)、[DELTA_PROJECTED](../DELTA_PROJECTED.md) |
| token/出力行でロードを共有 | 独立lane、列順の加算、端数fallback | [MATRIX_TAIL](../MATRIX_TAIL.md)、[COLUMN32](../COLUMN32.md) |
| 固定処理を準備時へ移す | 準備と推論の費用を別計上、モデル変更時の再準備 | [BORROWED_WEIGHTS](../BORROWED_WEIGHTS.md)、[PREPARED_ACTIVATION](../PREPARED_ACTIVATION.md) |
| 型付きcarryを直接送受信する | 進捗・入力identity、量子化粒度、F32部分和 | [DIRECT_MLP_REPLY](../DIRECT_MLP_REPLY.md)、[F32_COLUMN_CONTINUATION](../F32_COLUMN_CONTINUATION.md) |
| 演算を融合し、層境界をまたいで配分する | BF16丸め、通信/命令上限、失敗時の別journal | [TAIL_QUERY](../TAIL_QUERY.md)、[LIMIT_FALLBACK_REVIEW](../LIMIT_FALLBACK_REVIEW.md) |
| 出力に不要な終端演算を省く | 専用readoutの依存関係、保存すべきstate | [TERMINAL_READOUT](../TERMINAL_READOUT.md) |

Strassenなどの共有積は、変換・補正・ロードを含む費用で評価する。固定scaleの事前復元や大きな動的loopも、部分kernelの成功だけでは採用しない。不採用案は [COMPUTE_REDUCTION](../COMPUTE_REDUCTION.md) に整理されている。

全体bit一致、一般精度、命令、query数、通信、実時間、準備費用は別の指標として記録する。複数回の削減率を加算しない。queryとupdateを同じ料金・信頼性と扱わず、比較する実行方式を固定する。
