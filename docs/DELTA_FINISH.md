この文書の測定はDelta-finish v2採用時点。最新の整数kernelは[48-tokenでの重み共有](COLUMN16_TOKEN48.md)。

# Delta後半と出力投影を通常queryへ統合

2026-10-02。前半16-headのgated出力はclientで保持する。後半queryへその出力を添えて送り、後半16-headのgated出力をcanister内で直接つなぎ、全32-headの出力投影を実行する。後半のgated出力を返信して投影queryへ送り直す境界を除く。質問依存の状態はquery間でcanisterに保存しない。

## プロトコル

runtime/canister feature `experimental-delta-finish`とclient flag `--fuse-delta-finish`を明示する。既存`--fuse-delta-projected`が前提。

`delta_project_finish`はdims `[tokens,16,16,keep_recurrent_state]`、tensorは`<linear_attn>.in_proj_qkv.weight`、aux/scalarsなし。tokens1〜90、keepは0/1。入力は既存後半reuseの準備状態・conv履歴・初期F32再帰状態の後ろに、前半16-head gated出力をtoken-major `[tokens,16,128]`で追加する。前半出力は有限の正確なBF16値であることを検査する。

後半reuseを従来helperで計算し、前半・後半をtoken-major `[tokens,32,128]`へ組み立てる。元のINT8 out_proj、元F32 rank64 LoRA A/B、scale2、従来BF16境界で投影する。返信は投影結果 `[tokens,2560]`、後半conv履歴、keep=1の場合だけ後半F32再帰状態。前半の履歴・状態は先のqueryでclientが保存したものを維持する。

出力投影の固定tensor形状とdtypeを重み読み出し前に検証する。input logical900K・frame2MB・query5B・heap4GiBを維持し、返信の保守的なbyte上限を事前検査する。追加量子化や整数/F32の加算順序変更はない。

clientは16-headずつの2分割で、90 token以下かつcodec別の保守的入力byte上限が2MB以内の場合に統合する。古い`bf16-exact`はbitmapが大きくなるため、87-tokenの最悪入力では従来のreuse＋projectionへ戻す。132 tokenも従来経路を使う。フラグなしの経路は維持する。

## 部分検証

保存された16行tile版Wasmのprefix/主問題/情報不足/最大変更、第0/30層のreuse＋出力投影をoracleにした。候補の8通常queryは投影値・後半conv履歴・必要なF32状態の全値がbit一致した。新Wasmのheap cacheが空の状態で測定し、初期化・読み出し経路を含めた。最大handler2,671,563,465命令、query失敗なし。

無効なtoken/head/first/keep、前半出力の非BF16、準備INT8の末尾fractional/-128の7通常queryを拒否した。nativeは56 runtime tests＋2 integration tests＋2 compile-fail doctestsが通過。前半出力のNaN/Infも拒否する。clientは新経路・conv/状態保存・132-token fallback・古いcodecの容量fallbackを検査し、既存projected Deltaのテストも通過した。

部分記録`artifacts/delta-finish/partial/report.json`、build/sourceは`artifacts/delta-finish`。部分counterはcold重み読み出しを含み、旧oracleはfull cacheのため、部分counterの差を性能差として採用しない。全層比較はfresh heapから同じmanifest順に721 tensorを準備して実行する。

## 重複走査を除いた版

初回v1は主問題193→169、prefix194→170 query、通信も減り、全5条件で保持hidden/state/判断/確率がbit一致した。ただし主問題37,005,939、prefix321,754,159命令が増えた。v1は全面採用せず、ソースを`artifacts/delta-finish/v1-validated-source.zip`へ保存した。

wrapperの全入力有限性走査は、既存reuse helperが検査する準備状態・conv履歴・再帰状態を二度走査していた。v2は追加された前半gated出力だけをBF16/有限値検査し、残りを既存helperへ委ねる。両helperが有限値を保証した出力をコピーするだけなので、連結後の再走査も除いた。入力の拒否や数値演算を省いた変更ではない。

同じcold cache条件の部分比較で、prefixは1query17,742,374、主問題17,008,612、情報不足16,169,711、最大変更17,248,298命令を減らした。8出力は引き続きbit一致し、不正入力7件を拒否した。部分記録`artifacts/delta-finish-v2/partial/report.json`。

## v2の全層比較と採用

全5条件で保持hidden・状態・判断・確率が16行tile版の従来Wasmとbit一致した。失敗/replay0。prefixは全32層の全token hiddenと72状態配列を比較し、質問は31層の全token hidden＋終端層の最後のtoken、48保持状態配列を比較した。全32層・全tokenの演算を維持し、終端の不要状態省略は従来仕様のままである。

| 条件 | query | handler命令 | 削減命令 | 削減率 | Candid bytes | 実測秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 194→170 | 167,628,853,129 | 104,062,817 | 0.0620% | 237,812,108 | 24.0371 |
| 617 | 193→169 | 316,553,018,529 | 371,200,749 | 0.1171% | 354,192,982 | 39.8986 |
| insufficient | 193→169 | 271,114,723,951 | 345,358,301 | 0.1272% | 330,090,604 | 36.3325 |
| maximum | 193→169 | 320,933,148,062 | 378,259,941 | 0.1177% | 361,100,413 | 43.5753 |
| normal | 294→294 | 458,499,530,856 | 0 | 0.0000% | 580,794,410 | 61.6013 |

主問題は24 query（12.435%）、17,136,240 bytes（4.6148%）、371,200,749命令（0.1171%）を削減した。prefixも24 query・8,881,008 bytes・104,062,817命令を削減。初回prefix込み387→339 query。prefixなし132 tokenは294 query・命令・通信が同じでfallbackが働いた。全条件で命令が増えず、統合対象の4条件で減ったためv2を採用する。

主169・初回339・prefixなし294 queryで50/32未達。handler合計を5Bで割るだけでも主64・初回97 queryが下限であり、実際は通信・CDK・状態依存もある。50 queryへ到達したとはしない。

最大queryは主問題3,804,503,358、全条件3,860,151,610命令。最大観測heap4,119,986,176 bytes。選択したquery5B/heap4GiB上限を維持する。命令はCDK Candid encode/decodeを、通信はHTTP/CBOR/signatureを除外。heapはhandler終端page数で瞬間ピークではない。時間はquery cache未制御の単回測定で速度改善の保証ではない。

既存INT8版から追加劣化がないことの検証であり、公式BF16参照との差や最大変更gold=yesへのnoの誤判定は残る。型安全性と判断精度を別に報告する。

採用module `023bda911347f082b1f836105e817fab3480999f666135de4026a71bb5af6617`、専用local canister `4caro-hl777-77775-aaaba-cai`。`artifacts/delta-finish-v2-cache-checks/report.json`で38実装hash、認証module bookends、721 tensor・4,065,416,192 bytesのcacheと固定RoPE表131,072 bytesの不変を確認。検証時ソースは`artifacts/delta-finish-v2/validated-source.zip`、全層記録`artifacts/delta-finish-v2-*`、比較`docs/delta-finish-v2-summary.json`。

固定準備は別途721 update、15,041,482,761命令、233.5051秒、Candid request47,473/reply14,927,506 bytes。準備時pack status1/cache status2 query・認証read2、全層前後cache status2 query・認証read2は推論数から分ける。モデル変更/upgrade時の準備だけupdate、推論は通常query、中間状態はclient-heldを維持した。生成Wasm・測定JSON・中間状態はgitignore対象。

```sh
cargo build --offline --release --target wasm32-unknown-unknown -p imajev-inference \
  --features experimental-projection-reuse,experimental-full-weight-cache,experimental-blake3,experimental-attention-fusion,experimental-delta-projected,experimental-prepared-rope,experimental-column16,experimental-delta-finish
# 専用local canisterへinstall/upgrade後:
.venv/bin/python scripts/prepare_weight_cache.py --canister <local-id> \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --directory artifacts/new-preparation --include-f32 --require-prepared-rope
.venv/bin/python scripts/validate_prepared_weights.py --canister <local-id> \
  --run-name new-finish-validation --baseline column16-v1 \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --preparation artifacts/new-preparation --reuse-projection-inputs \
  --frame-checksum blake3 --fuse-attention --fuse-delta-projected \
  --fuse-delta-finish --require-prepared-rope
```

