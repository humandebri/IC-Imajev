# 用語と名前

| 名前 | 意味 |
|---|---|
| inference-core | モデル非依存のF32投影と可逆activation codecを実装するcrate |
| imajev-runtime | Imajevの演算・数値境界・モデル固有op・frame dispatchを実装するcrate |
| inference canister | 重み準備・アクセス制御・query入口・固定重みcacheを担当するICアダプタ |
| client scheduler | 演算の依存順序、query分割、carry保持、再開と上限時の切替を担当する処理 |
| prepared weights | 固定重みを準備時に復元・検証・配置変換し、推論で借用する表現 |
| quantized input | blockごとの整数値とscaleを保持した検査済み入力。繰り返し量子化しない |
| carry | 分割演算を継続するための部分和・整数入力・scale・残差・進捗情報 |
| exact | 指定backend・参照・量子化契約のbitを保存すること。公式モデルや別backendとの一致を含意しない |
| handler instructions | handler内の命令counter。Candid入口外処理や通信処理の費用と区別する |

既存crate名、op名、encoding名、Cargo feature名は互換性のため維持する。新しい共通crateだけモデル名を外した名前にする。`experimental-*` は検証条件や選択経路を表すため、一括改名して安定APIとは扱わない。
