# 毎queryのDEFLATE展開を省く可逆carry

2026-10-03。MLP→Delta融合の診断でDEFLATE展開に約1.6億命令がかかったため、`mlp-delta-huffman-carry-exact-v1` を追加した。旧zlib protocolは維持し、実験用の別encodingとして比較する。モデル演算・INT8入力・F32 LoRA・BF16丸めは同じ。

carryの6種類のデータを15 byte planeへ分け、planeごとにraw、constant、canonical Huffmanを選ぶ。descriptorはmode u8 / payload bytes u32、Huffmanだけ256 code lengthsを付ける。高entropy planeはraw、定数planeは1 byteで運ぶ。クライアントはcanister生成済み値のbyteを圧縮するだけで推論しない。queryは辞書・LZ距離・Adler展開処理を省き、10-bit tableと最大24-bit fallbackで復号する。frame全体の既存digest検証は維持する。

出力長は検証済みshapeから固定し、圧縮descriptorから確保しない。過剰・不完全code tree、末尾byte、非zero padding、short payload、不正mode/lengthを拒否する。復号後は元の型検証とrequest identity照合を行う。質問状態はクライアント保持のまま、固定モデルの準備だけupdateを使う。

## byte一致と容量

実際の87-token / prefix45 / partial down256行のcarryを層0・1・3で比較。新Python encoder→Rust decoderの6 groupすべてが、保存済み旧DEFLATE requestの展開結果とbyte一致。Rustのtyped request decodeも通過。

|層|旧zlib frame bytes|新Huffman frame bytes|
|---|---:|---:|
|0|1,821,966|1,733,969|
|1|1,831,419|1,744,197|
|3|1,835,210|1,752,857|

新frameは保存済み正式request headerとcanister生成済みcarryから再現する。上記は容量・可逆性の検証で、命令削減やquery完走の証明ではない。

候補の全runtime flagsで85 unit通過（実fixture用1 testはignore指定）、旧・新encoding両方のidentity/不正値integration4件通過。実fixtureのignore testはBLAKE3 featureを有効にして別途実行・通過。version2 fixtureへBLAKE3 featureなしでdecodeした初回テストはunsupported frame versionで失敗し、正しいfeatureで再実行した。生成物はartifactsへ保存しignore。

## 実測

専用実験canister `6eydd-o3777-77775-aaama-cai` だけを更新。module `05c592756eac051de6d2c4733ff86b204557b3de16b17d29830320f44539fcef`。固定重み準備後、同一moduleの分離MLP/Deltaと融合結果を比較した。87-tokenのpartial prepare→finishは通常MLPと全bit一致。q/scales/Aは保存済み旧状態とbit一致。87-token融合は3層とも5B上限超過、未採用。45-tokenの融合診断は分離hidden/Delta/convと全bit一致。21通常query（87-token失敗3を含む）。45-tokenを本番87-tokenの代用にしない。

|層|45-token融合命令|wire decode|うちplane復号|旧zlib版から融合削減|87-token partial finish命令|
|---|---:|---:|---:|---:|---:|
|0|2,846,362,043|207,564,496|128,851,670|34,884,907|1,327,100,952|
|1|2,846,711,685|207,647,003|128,893,203|30,034,496|1,327,796,611|
|3|2,847,223,007|207,976,281|129,015,776|29,533,930|1,328,305,160|

nested counterはinclusive。45-token復号は約3046〜3477万命令減（約12.8〜14.3%）。87-tokenのpartial finishも約3406〜3657万命令減った。ただし全graphには新codecをまだ組み込まない。byteごとのHuffman loopが約1.29億命令残る。次は複数symbolのtable復号で繰り返す分岐・bit取出しを減らす候補を測る。50/32 queryは未達。

固定cacheは721 tensors / 4,065,416,192 bytes。主canisterのcertified module照合は従来hashと一致。Layaへの操作は行っていない。融合の証拠は `artifacts/prefix_codec/carry-huffman-fusion-check/report.json` / `validated-source.zip`。

## 再現

```sh
.venv/bin/python scripts/check_carry_planes_native.py
cargo test --offline -p imajev-runtime --features experimental-mlp-delta-fusion --lib
cargo test --offline -p imajev-runtime --features experimental-mlp-delta-fusion --test mlp_delta_carry_identity
CARRY_FIXTURE_ROOT="$PWD/artifacts/prefix_codec" cargo test --offline -p imajev-runtime --features experimental-mlp-delta-fusion,experimental-blake3 python_huffman_matches_recorded_deflate_bytes -- --ignored
.venv/bin/python scripts/check_mlp_delta_fusion.py --huffman --canister <experimental-id> --wasm artifacts/<build>/full.wasm --directory artifacts/<fresh-report>
```

buildとpatchは[MLP_DELTA_FUSION.md](MLP_DELTA_FUSION.md)と同じflags。証拠は `artifacts/prefix_codec/full-build-carry-huffman/source.zip` / `source-hashes.json` / `patch.json`、`artifacts/carry-huffman-full-unit-v2.log` / `carry-huffman-identity-v2.log` / `carry-huffman-fixture-v2.log`。runtime/clientを測定中に編集しない。

byte fixtureの再現reportは `artifacts/prefix_codec/carry-huffman-native-parity.json`、主canister照合は `main-unchanged-carry-huffman/report.json`。

## 既存経路の回帰と返信shapeの修正

測定moduleで全5条件（旧log主問題も加えた6経路）を完走。全32層のhidden/state、判断4条件のvalue/abstain/raw logits/probabilities/unknown probability/最終hiddenが保存済みINT8版とbit一致。typed output有効、失敗/replay0。主87-tokenは64 query / 260,701,701,042命令 / 124,634,637 Candid bytes / 最大query4,555,651,098命令 / 30.280秒（単回）。新codecをgraphへ採用した数字ではない。既存の判断の見逃しは残る。

証拠は `artifacts/prefix_codec/full-carry-huffman-base-proof/report.json` / `validated-source.zip`。prefix packetは2回目の準備query/命令/通信が0（`codec-second.json`）。現行完成graphのquery数は64で、50/32未達。

回帰測定後、融合返信が87/89-tokenで45万floatを超えることをコード確認で発見した。命令上限を通った場合でも汎用encodeの45万float boundが返信を拒否する不具合だった。新旧carry encodingに限り、metadataから導いた厳密返信数 `2*n*2560+24576` を使うよう修正した（最大480256 float）。通常codecと900K float上限は変更しない。正しい87/89-token返信のencode/decode bit一致、1 float不足、不正token数、通常codecへの同じ配列の拒否をintegrationに追加、全5件通過。

この返信修正は上記測定moduleの後の変更であり、融合が5B内で完走した証明ではない。修正後のWasmは `artifacts/prefix_codec/full-build-carry-huffman-reply-shape` に別途build/patchし、主・実験canisterへはinstallしない。次の復号改善と合わせて再測定する。
