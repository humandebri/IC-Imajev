# 2026-10-04 実装レビューとMLP前倒しの境界

主問題はprefix45 + suffix87 = 132 token、rotations=1を維持する。固定モデル・base INT8 block256・元F32 LoRA・BF16境界は変更しない。新APIは通常queryで進め、状態はクライアントに保持する。全体の現行連結経路は51queryであり、50/32到達を主張しない。

## 修正した問題

連結モードは`--roll-begin`・`--roll-heads`・`--roll-down`の非既定値をsessionに記録する一方、実際は固定幅を使っていた。指定が実行条件を表さないため、`JoinedPrefixGraph`の初期化前に非既定値を拒否する。既定値は互換性のため維持し、実行時の固定幅は`JOINED_QUERY.md`に記録した構成を使う。境界テストで、拒否前に基底graphを初期化しないことを確認する。

profileスクリプトが計測後に参照hashを取得していたため、実行前に要求・期待返信bytesを固定し、その固定コピーをbridgeへ渡すよう修正した。終了時に参照・ソース・Wasm・moduleを再照合する。Python `-O`と空でない証跡ディレクトリを拒否し、canister/Cargo/model/manifest/proof helperもソース固定に含める。対象query番号を指定でき、最初の数件だけでなく後段のボトルネックを測れる。

新codecのRust/Python境界・方向・長さ・有限値・scale・整数・進捗・層scopeをレビューした。prefix圧縮は2/4 byteのplaneを可逆復元し、既存の未圧縮decoderへ渡して同じ検査を適用する。型付き返信はruntime内で構築したcarryのみを包み、request全フィールドと次stepに結び付ける。追加量子化は行わない。

## 実装した前倒しAPI

- `mlp_finish_attention_mlp_front`: 部分MLP downを完了し、次Attention全体と次MLP前半を同じqueryで計算する。返信は前MLP hidden + 型付きMLP carry + KV。重複したnorm/Attention出力の返信を省く。
- `delta_partial_mlp_front`: Delta残りheadsと出力投影を完了し、次MLP前半へ直接進む。
- continuationのtag10: 従来のF32 base可逆圧縮に加え、prefix conv/key/innovation/gatesも可逆plane圧縮する。旧tagと既定動作は維持する。

計測用`check_attention_front_chain.py`は、独立した参照入力から作る単体controlと、実際に連結したqueryを区別する。参照bytes・NPZ・ソース・Wasm・moduleを固定し、失敗requestも別名で保存する。部分経路成功は全体query削減とは扱わない。

## 主87 tokenの実測

実験canisterは`6eydd-o3777-77775-aaama-cai`、moduleは`87118d34eb405b05d3c7024b05da2ff833695cb0ad23df40842777c514d7b272`。主canister・Layaは変更していない。固定重み準備は721 update、78,739,929,194命令、4,065,416,192 bytes。

| MLP前半 | MLP22 finish + Attention23 + MLP23 front | MLP23 complete + Delta24 8 heads |
|---|---:|---:|
| 256行 | 4,849,783,894 | 4,854,901,593 |
| 384行 | 4,888,589,726 | 4,820,016,901 |
| 512行 | 実query 5B超過 | 未実行 |
| 640行 | 実query 5B超過 | 未実行 |

成功した部分経路のhidden・MLP carry・KV・選択head convは既存INT8参照とbit一致。8 headsでは次のDelta残り+MLP front要求が2MBを超える。6 headsも同様。10/12/14 headsへ増やすと前queryが5Bを超える。12組の要求された三連結は完了0。幅だけを変える案は現状採用できない。

証跡は`artifacts/attention_front/review-chain-v2`〜`v4`。上限超過はIC0522、frame超過は送信前に拒否し記録した。handler命令はCDKの外側Candid処理を除くため、実queryの成功も併せて確認する。

8 headsの次要求payloadは2,099,946 bytesで、header/hashを含む前から上限を超える。F32 baseは890,880 bytes、既存plane Huffmanで746,890 bytes。zlib・横方向XOR・token方向XOR・dictionaryを同じ実データで試したが縮小しなかった。`base-compression-study.json`・`frame-compression-study.json`はartifacts配下に保存した。圧縮方法を替えるだけの採用根拠は得られていない。

80 tokenの情報不足では`review-chain-info-v1`で三連結が完走した。各queryは4,420,300,768 / 4,410,120,393 / 4,055,025,686命令、合計12,885,446,847命令。最後の要求1,979,483 Candid bytes、返信1,132,554 bytes。全carry・hidden・KV・convは独立controlとbit一致。これは部分経路であり、主132 token全体の削減を意味しない。

## 検証

Rustは採用feature構成で144 unit（1 ignored）・15 integration・11 doc、直接返信なしの最小featureも通過。Pythonの新codec3・既存codec11・連結境界3テストが通過。再開journal6・参照固定5・roll境界3・連結codec6も通過。`profile_roll_graph.py`は構文/CLI検査が通り、Python `-O`を拒否した。

`artifacts/attention_front/review-full-v1`の標準6条件（prefix、prefix継続の元主問題、主、情報不足、最大変更、prefixなし）は全層hidden/state/最終hidden/判断bit一致。ソース88 filesと参照329 filesを固定し、ソースarchiveを保存した。

連結3条件も`artifacts/attention_front/review-joined-v1`で全出力bit一致、失敗/replay0。主51queryは240,785,289,592命令・150,311,010 Candid bytes・最大4,918,963,064命令・観測heap4,144,037,888 bytes・単回39.582秒。情報不足51、最大変更は安全な62query経路を維持した。前回51query比で33,031,279命令（約0.0137%）減、通信不変。これは重複する有限値走査などを除いた小さい改善で、全体50/32や一般的な速度向上を意味しない。元BF16との差と既存の最大変更の見逃しはこのbit一致検証の対象とは別で、解消を主張しない。

## 続行したボトルネック計測

修正したprofileスクリプトで、連結37/38番と標準46/47番の実4queryを再実行し、全出力bit一致、参照固定と終了照合を確認した。証跡は`artifacts/attention_front/review-profile-v2`。計測専用queryは本推論回数に加えない。

MLP23は約4.054Bのうち整数base投影3.159B（77.9%）、LoRA A合計約198.91M、B約320.40M、wire decode/encode約7.57M。wireだけ削っても50query化の主要な余裕を得られない。Attention23は約4.021Bのうちbase約1.645B（40.9%）、LoRA A/B約359.24M、wire約7.09M。残りのnorm/RoPE/GQAなどはこのprofileでは分離されていないため、全てGQAの費用と断定しない。

実装を読むと、GQAはheadごとに同じKVを再コピーし、`execute`のmetadata/有限値検証を繰り返している。QKの内積も256要素を順番に加算するscalar経路。共有KVをsliceで渡す経路と、積だけSIMD化し加算順序を維持する経路を次の診断候補にする。値のSIMD加算は既に実装済みで、重複した案として数えない。

## 次の削減

最後のDelta残りとterminal tailの単純融合は合計約6.12Bで実行できない。最後の12queryの合計は約54.60Bであり、11query化には前のqueryへ処理を移す必要がある。ただし今回の前倒しには、命令上限とF32部分和の通信上限が同時にある。まずF32部分和を再送しない経路、または整数base kernelで反復するロード・offset計算の削減を個別profileで評価する。元F32加算順序や量子化境界を変える案はbit一致を別途検証する。
