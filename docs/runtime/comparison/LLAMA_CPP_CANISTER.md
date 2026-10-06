# llama_cpp_canisterとの効率比較

2026-10-05。**準備済みINT8投影のIC命令数ではImajevが有利だった。モデル全体の優劣は未確認。** 合成整数入力の10形状をlocal ICで比較し、38.5〜79.4%の命令削減を確認した。[実測条件と結果](KERNEL_BENCHMARK.md)。比較するモデル、量子化、処理内容が異なるため、この差を推論全体の速度倍率には使えない。

比較先は [onicai/llama_cpp_canister](https://github.com/onicai/llama_cpp_canister/tree/e987fca5ff91dbff3026a1ac5ff44cb372039212) のcommit `e987fca5ff91dbff3026a1ac5ff44cb372039212`。参照するllama.cpp forkは `91c63a2284ff99c5761a27c3865b312fb8eca148`。取得ファイルのSHA256とURLは [sources.json](sources.json) に保存した。外部canisterへの推論要求・deploy・変更は行っていない。

## 実装と測定対象

| 項目 | Imajev | onicai |
|---|---|---|
| モデルと出力 | Qwen3.5-4B系＋F32 LoRA＋専用判断head | GGUFモデルの会話生成 |
| 重み/入力 | INT8 base、block256入力、既存BF16境界 | モデルごとにGGUF量子化を選択 |
| 分割の状態 | carryをクライアントが保持 | canister内contextとsessionファイルを利用 |
| 実行 | 通常query。固定50要求のreplicated ingress再送も計測 | query/updateの両API。READMEの生成手順・主要実測はupdate |
| 融合・再開 | モデル固有graphとjournal | llama.cppのdecodeとprompt cache |

相手のCandidには `run_query` と `run_update` がある。[固定したCandid](https://github.com/onicai/llama_cpp_canister/blob/e987fca5ff91dbff3026a1ac5ff44cb372039212/src/llama_cpp.did)。contextの保持とsession読込/保存は [main_.cpp](https://github.com/onicai/llama_cpp_canister/blob/e987fca5ff91dbff3026a1ac5ff44cb372039212/src/main_.cpp) で確認した。

相手のwrapperは各runで `llama_memory_clear(mem, true)` を呼び、その後sessionを復元できる構成になっている。KV状態の初期化・復元も総費用の比較対象になる。[固定したmain_.cpp](https://github.com/onicai/llama_cpp_canister/blob/e987fca5ff91dbff3026a1ac5ff44cb372039212/src/main_.cpp)。今回の投影測定にはこの処理を含めていない。

## 現在の数値

こちらの固定要求再送では、prefix45 token準備済み・suffix87 tokenの判断に50呼び出し、234,521,052,335 handler命令、149,353,220 Candid bytesを要した。localの呼び出し待ち時間合計はquery37.52秒、replicated ingress53.30秒。後続入力は保存したchainから取得しており、新しい返信からgraphを組み立てるend-to-end計測ではない。prefix・固定重み準備も集計外。[REPLICATED_INFERENCE](../../REPLICATED_INFERENCE.md)。

相手のREADMEはQwen3-1.7Bの生成を約6.4B命令/tokenと報告し、context4096の最初のtoken前の費用を1.4B命令としている。これは生成のmainnet計測で、こちらのsuffix一括処理と対応しない。[固定したREADME Appendix A](https://github.com/onicai/llama_cpp_canister/blob/e987fca5ff91dbff3026a1ac5ff44cb372039212/README.md#appendix-a-max_tokens)。

234.52Bを87で割って生成tokenあたり費用と呼んだり、6.4Bとの比を高速化率としたりはできない。パラメータ数で割るだけでも、Attention/Deltaの構成、LoRA、語彙head、batch処理、量子化方式の違いは補正できない。

## 優位になり得る部分と交換条件

固定重みの事前配置、量子化・LoRA Aの共有、専用判断head、client-held carryによる融合は、こちらの用途で有利になる候補である。一方、こちらは約149MBの中間通信を必要とし、固定cacheだけでも約4.065GBを使う。queryの安さだけで、継続的な会話生成・メモリ効率・on-chain利用の信頼性まで優位とは扱わない。

演算の第一段階は合成入力で実施した。共通runtimeの第二モデルにはLayaを使う。[検証方針](../ARCHITECTURE.md#第二モデルはlayaで検証する)。次は通常の量子化入力の精度と準備費用を調べる。相手との推論全体のA/Bには、双方で実行できる同一モデルが別途必要になる。

1. **演算比較**：同じ整数operand、shape、scale、加算/丸め契約で線形投影を実行する。Wasm命令、準備費用、heap、出力差を記録する。量子化blockの異なるkernelは変換費用と精度差も含める。
2. **推論比較**：同じモデル・重み・token IDs・context・prefill長・生成長・cache状態・実行方式・replicaで測る。prefillとdecodeを分離し、cold/warm、総命令、通信、cycles、時間分布、失敗、品質を比較する。

双方のcounterの範囲も揃える。共通モデル/量子化を実装できない場合は用途比較として報告し、runtimeの速度倍率を出さない。IC向けの投影kernelを汎用化する根拠は得られたが、推論全体の優劣は未測定である。
