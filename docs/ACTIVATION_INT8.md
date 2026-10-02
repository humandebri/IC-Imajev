# INT8通信とquery融合の実測

2026-10-01。重みは既存のINT8 base packに固定し、F32 LoRA・専用readout・calibration・tokenizerを変更していない。中間activationの通信量子化と、重み量子化を別に評価する。演算本体は既存のF32積和と公式境界のBF16丸めを使う。整数SIMD積和は別の実験primitiveであり、全モデルの演算を整数へ置換した結果ではない。

BOOM DAO 617、132 token、1質問、rotations=1のactivation INT8版は全32層から専用readoutまで通常queryで完走した。3,108 query、Candid request+reply 1,284,768,155 bytes、514.639秒。可逆BF16通信版の2,549,073,399 bytesから49.60%減。判断は双方 `likely` だったが、候補確率の最大差は0.120719、logitの最大差は0.660635であり、精度同等とは言えない。公式非量子化参照との差には重みINT8・演算差・通信INT8のすべてが含まれる。[各queryと精度比較](activation-int8-results.json)。選択肢の別順序や情報不足の全層INT8評価は未実施。

## 通信形式

`encoding="int8-block256-v1"` のpayloadは `u32 count | u32 quantized_prefix | blocks | F32 tail`。各blockはF32 scaleと最大256個のINT8値。最大絶対値/127のscale、round-to-nearest ties-to-even、範囲[-127,127]。零blockはscale=1、微小値では最小正F32までscaleを下限化する。token ID、DeltaNetのdecay/beta、recurrent stateはF32のまま運ぶ。stateをINT8化した測定ではない。

SHA256、shape、count、prefix、finite、正のscale、reserved -128、完全長、model/pack/stepを検証する。[実canisterでの不正状態拒否](int8-wire-contracts.json)。F32 tailの輸送は可逆でも、stateの演算入力となるactivationが変わるため、BF16通信版とstateの数値は変わり得る。

## head融合

`delta_heads_bf16` は複数headを1 queryで評価する。入力はheadごとのq/k/vを並べたINT8 prefix、その後に各headのdecay/beta/F32 state。返信はheadごとのoutput prefix、その後にF32 state。dimsは `[tokens,key_dim,value_dim,heads]`、最大16 heads。INT8のこのopに限り入力要素上限を1,200,000へ拡大し、encoded blobの2,000,000 bytes上限を維持する。132 tokenの16 headsは要求1,889,568 bytes、返信1,323,628 bytesだった。

DeltaNet内部はstateを一時的にtransposeし、独立する4 valueをSIMD laneに載せる。各valueのkey積和順はscalarと同じ。旧16 queryの35.373億handler命令を、融合1 queryの14.348億へ減らした。実入力でnative/Wasm、旧queryのoutput/stateがすべてbit一致した。[融合の実測](delta-fusion-check.json)。stateはquery終了後クライアントへ返し、canisterへ永続化しない。

`attention_heads_bf16` は最大8 headsのcausal attention、`rope_heads` は最大16 headsのRoPEをまとめる。どちらも各headのscalar演算順を保つ。INT8のblock境界はpayload順に依存するため、別のtoken長・tile幅についてbit一致や判断精度を一般化しない。

全32層を通す融合版は **2,132 query、461.531秒、Candid通信1,283,895,563 bytes、最大query命令数3,022,101,308、終端heap最大観測86,441,984 bytes**。融合前3,108回から976回（31.4%）削減した。全32層のhidden、conv/DeltaNet/KVの保存状態、final logits・候補確率・unknownが融合前INT8通信版とbit一致した。型付き出力も有効で、判断はlikely。ただしBF16通信版との精度差は残る。[全query実測と比較](fused-results.json)。通信削減はhead融合単独で約0.068%であり、query数ほど減っていない。

後続のLayaレポート比較で射影ループを改善し、総命令数をさらに25.66%減らした。query数2,132/通信1.284 GBは同じ、単回251.002秒。全層/state/判断のbit一致。最新の原因分析と実測は [LAYA_COST_ANALYSIS.md](LAYA_COST_ANALYSIS.md)。

## 32 query目標の制約

通常queryの現在の上限は1回50億命令。[ICP公式のresource limits](https://docs.internetcomputer.org/references/resource-limits/)。head融合時点の1層は約1,000億handler命令、後続の射影改善版も約714〜733億handler命令である。1層を1 queryへまとめるだけでは上限に入らない。クライアントの外側だけ32回に見せたり、推論をupdateへ移したり、途中stateをcanisterへ保存したりしていない。

整数SIMDの別primitive `int8_matmul` は、activationをtokenごと256列blockに量子化し、INT8 weightをi16へ展開、8 tokens×8 output rowsのI32 dotを使う。132 tokens×768 rows×2560 colsの実QKV tileは6.2899億handler命令、native/Wasm bit一致だった。[prototype測定](integer-kernel-check.json)。このtile測定から8192行へ線形に外挿するとbase QKVだけで約67億命令になる。これは全QKVの実queryでも、LoRA込みの判断精度検証でもなく、理論的な最適実装の下限でもない。

32前後の通常queryへ収めるには、このprototype以上のdense演算削減と層内融合が必要。モデルの蒸留・小型化は精度を再評価する別案。ローカル専用の命令上限緩和は通常ICへ移せない別条件であり、今回のnetwork設定は変更していない。現行4Bの32 query達成は未達として扱う。

## 再現

既存の専用canisterは `46el7-ql777-77775-aaada-cai`、endpointは `http://localhost:8001/`。seal済みpackを保持するupgrade後に、保存済みsessionと異なる新規directoryで実行する。

```sh
cargo test --workspace --offline
cargo build --release --target wasm32-unknown-unknown -p imajev-inference --offline
cargo build --release -p imajev-runtime -p imajev-client --offline
.venv/bin/python scripts/test_int8_wire.py
.venv/bin/python scripts/test_head_fusion.py
icp canister install 46el7-ql777-77775-aaada-cai --mode upgrade \
  --wasm target/wasm32-unknown-unknown/release/imajev_inference.wasm \
  --network local --identity imajev-local
.venv/bin/python scripts/run_full_canister.py --wire-codec int8-block256-v1 \
  --delta-head-cap 16 --attention-head-cap 8 --directory artifacts/new-fused-run
.venv/bin/python scripts/check_full_native_wasm.py --directory artifacts/new-fused-run
```

時間は単回local測定、cache/warmup/ホスト負荷未制御。命令数はhandler counterでCandid decode/encodeを含まない。通信はCandidだけ、HTTP/CBOR/署名を含まない。heapはhandler終端の最大観測値。型付きchoiceの有効性と判断のgold一致は別々に報告する。
