# 終端Delta状態の返送省略とGQA共有

2026-10-02。通常queryによるINT8推論に `--compact-heads` を追加した。演算、量子化、LoRA、専用readout、calibrationは変えず、不要な返信と重複したK/V入力を減らす。中間状態は引き続きclientに保持する。

132-token BOOM DAO 617、rotations=1の全32層を専用local canister `4caro-hl777-77775-aaaba-cai`（8001）で実測した。

| 指標 | 通常実行：前→後 | prefix準備済み：前→後 |
| --- | ---: | ---: |
| query | 708→**700** | 611→**603** |
| handler命令 | 711,017,272,458→**703,573,958,783** | 492,550,510,942→**485,354,469,101** |
| Candid bytes | 1,247,760,195→**1,182,062,887** | 872,750,670→**807,053,362** |
| 単回秒 | 87.311→95.997 | 99.448→64.249 |
| 最大query命令 | 4,395,987,402→同じ | 4,511,178,558→同じ |
| handler終端heap最大観測bytes | 41,811,968→同じ | 41,287,680→同じ |

命令削減は通常1.047%、prefix利用1.461%。通信は通常5.265%、prefix利用7.528%減。命令の主因である整数dotは同じなので、通信ほど大きな命令削減にはならない。時間は条件を統制した反復比較ではなく、通常実行では増えたため速度向上を主張しない。32/50 queryは未達。

## 変更と互換性

- `gqa_heads_bf16`：16 query headsに対して4組のK/Vだけを送る。132 tokensでは8層のAttentionが各2→1 query。147 tokensでは12+4 heads、512 tokensでは2 headsずつに分割し、blob・float数・演算量上限を守る。各headのcausal積和順は従来と同じ。
- `delta_terminal_bf16`：最後のtoken chunkでは出力だけを返し、終端のF32 recurrent stateを返さない。途中のchunkは状態を返す。prefix準備では終端状態も必ず返す。
- terminal実行の保存stateはconv/KVのみとなり、24個のDelta state配列を含まない。継続用cacheの代わりには使えない。未完了queryを再開するためのclient journalと入力は保存する。
- 両opはlossless BF16通信限定。Candidのメソッド追加はなく、既存 `step(blob)` を使う。新設定をsession identityへ含め、既存journalの混在を拒否する。既定経路は維持し、新しいflagで有効化する。

初期ゼロstateの入力省略、prefix Attentionの不要query位置の省略、回転量子化は今回の実装に含めない。探索時の通常通信92.4 MBという省略候補のうち、今回は約65.7 MBを実装した。

## prefixの準備費用

45-token prefix準備は578→**570 query**、255,695,466,238→**255,259,104,377命令**、463,530,574→**458,823,246 bytes**。単回46.516秒。全32層hiddenと72個のstate配列が旧cache生成とbit一致。

新しいWasm・graph source hashに対してcacheを実際に生成し直した。旧cacheのhashを書き換えて流用していない。617の最初の質問は準備込み**1,173 query・740,613,573,478命令・1,265,876,608 bytes**となり、通常実行より高い。

## 検証と制約

Rust25件、Python32件を通過。Python検証には、途中chunkのstateが次chunkへ渡ること、最終chunkだけ省略すること、prefix準備で保持すること、GQAの末尾groupと同じKVを共有する対応関係を含む。

実canisterで通常/prefixの非ゼロ初期状態を含むDeltaNet、GQA16 heads、147-token/12 heads、512-token/2 headsを旧演算と比較し、bit一致した。512-token試験はhead演算の境界検証であり、全512-tokenモデル推論ではない。512-token/2 headsの実測は3,764,257,817 handler命令。

全層では各入力の32層hidden、保持した48個のstate配列、最終hidden、raw logits、全確率、unknown、判定を比較する。省略した終端state自体は全層比較の対象に含めない。元のINT8方式への追加誤差を検査しており、公式未量子化モデルへの精度同等性ではない。型安全性とgoldへの判断精度も区別する。

3入力すべてで上記のbit一致、型付き出力の妥当性、journal replay 0を確認した。

| prefix利用入力 | 全token／処理suffix | query | handler命令 | Candid bytes | 単回秒 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 617 | 132／87 | 603 | 485,354,469,101 | 807,053,362 | 64.249 | likely |
| 情報不足 | 125／80 | 571 | 425,569,802,943 | 733,215,214 | 80.108 | unknown |
| 重大変更 | 134／89 | 603 | 500,935,345,142 | 824,160,673 | 63.469 | no（gold yesで既存の誤判定） |

重大変更の見逃しは改善していない。今回の数値保存と判断精度の改善を同一視しない。

命令数はCDK Candid decode/encodeを除いたhandler counter、通信はHTTP/CBOR/署名を除くCandid request＋reply。heapは終端page数で瞬間peakではない。prefix準備をhit費用へ含めない。各queryの記録と比較結果は `docs/compact-*-results.json`、raw journalは `artifacts/compact-*` に保存し、生成物はgitignore対象。

## 再現

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
# 固定packを準備済みの専用local canisterへupgradeした後
.venv/bin/python scripts/check_compact_heads.py --canister <id>
.venv/bin/python scripts/validate_compact_graph.py --canister <id>
```

全層検証スクリプトは既存の旧版journalを比較対象とする。通常の新しい推論では `run_full_canister.py` / `run_prefix_canister.py` の従来の整数・lossless・融合設定に `--compact-heads` を追加する。prefix準備と継続は別々のdirectoryを使う。新しい試行には新規directoryを指定し、測定済みjournalのreplayを性能測定へ混ぜない。

採用Wasm SHA256: `06078c66268ca498f98ffc0e556bdf35bab5719b3d40721ce3061d0e679b7951`。Laya環境は変更していない。mainnet、push、PRは実施していない。
