# 分割query間の量子化・LoRA A投影の再利用

2026-10-02。固定データの準備だけでなく、同じ入力をrow分割で投影する際の重複を除く実験を実装した。`experimental-projection-reuse`は既定では無効。最初の`lora_integer_capture`が通常の出力tileに続けて量子化済み入力・元F32 block scale・元F32 LoRA A投影を返し、次の`lora_integer_reuse`がその状態からbase整数dotと元F32 LoRA B投影を行う。専用の準備queryは増やさず、2通常queryのままとする。中間状態はclient-heldで、updateで質問に依存する計算を行わない。

元モデルのW8/base、block256量子化、元adapterのF32・BF16加算境界は同じ。受信状態を信用せず、integer値の範囲[-127,127]・整数性、scaleの正の有限性、A結果の有限性、shape/range/work/frame上限を検証する。opaque `QuantizedRows`の再構築後は既存の安全条件を使う。状態の整数値をF32に直して運ぶ初期版に加え、検証・復元・返送用整数→F32変換をSIMD化した版を測定した。既存公開reader APIと通常演算は維持する。

原型はモデルlayer0のQKV投影8192×2560・rank64。全層保存実行の132-token入力から1/7/8/32/64/87/132 rowsを使い、通常2tileとcapture/reuse2tileを比較した。両方式とも4096+4096行とし、変更した境界の結果を独立した旧native scalarと比較した。実モデルの重みを再量子化せず、base/A/Bだけを23,756,800-byteの別packへコピーした（SHA256 `49f573902ad63335c4c69ede0d5ecea21b2857e0673c0e2d814c5772bcac4a04`）。元packの識別子とコピーしたtensorのoffset・内容hashをmanifestへ記録した。

| tokens | 通常2queryの命令数 | SIMD reuse 2query | 命令削減率 | 通常Candid bytes | reuse Candid bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 66,726,197 | 60,734,273 | 8.980% | 29,278 | 35,020 |
| 7 | 271,466,011 | 253,345,512 | 6.675% | 189,064 | 229,087 |
| 8 | 177,453,787 | 178,366,638 | -0.514% | 215,694 | 261,430 |
| 32 | 646,291,719 | 653,761,098 | -1.156% | 854,830 | 1,037,686 |
| 64 | 1,273,268,545 | 1,289,908,807 | -1.307% | 1,707,006 | 2,072,688 |
| 87 | 1,859,367,495 | 1,861,824,383 | -0.132% | 2,319,508 | 2,816,595 |
| 132 | 2,672,025,432 | 2,706,364,623 | -1.285% | 3,517,884 | 4,272,071 |

初期版は132-token warmで通常2,671,983,342→候補2,751,219,579命令（2.965%増）。SIMD化後は通常2,672,025,432→候補2,706,364,623命令（1.285%増）。132-token coldでは2,739,590,414→2,767,810,459命令（1.030%増）。1/7-tokenでは改善したが、主要入力132-tokenで通常版より重いため全モデル採用は保留する。通信も3,517,884→4,272,071 bytes（21.44%増）。この部分実験の短い入力での勝利を全モデル高速化として扱わない。

各版で7形状×cold/warm/clear×4通常query=84queryが完了し、全出力が独立旧nativeとbit一致。計168成功query。SIMD版では別途4通常queryで-128・128・小数0.5・zero scaleを送り、全て`prepared activation wire bounds`で拒否した。これらは成功query数・性能集計に含めない。native fixture44＋compile-fail doctest1、canister cache3 testsが通過。通常feature側はruntime43＋cache3＋doctest1が通過した。

準備はcold clear、warm base/A/Bの3update、cleared clearの計5update。uploadの17updateと分けて記録した。各warm queryのstable readは0。cold時にはreuseがA重み655,360 bytesの読み出しを省く。queryのhandler命令はCandid decode/encodeを除き、通信はCandid request+replyだけ。単回時間・終端heapは生reportに保存し、普遍的な速度や割当peakと解釈しない。SIMD版cold/cleared132-tokenのhandler終端heapは472pages（30,932,992 bytes）。全モデルの判断精度・校正・query数は未検証。

新しい専用local canister `4qggx-l3777-77775-aaaca-cai`で試験し、主canister・既存診断・Layaを変更しなかった。初期module `ab420f8d4791eeec2e9973def5ceb0eaaf8434056d640eed30956e60c3987222`、SIMD版 `3365a07e50cee02f51536dab2252741733a1b0e311c06356e80be8f08401b5f6`を試験前後に認証済みhashで確認した。SIMD版を残した専用canisterは次の試験にも使用できる。主module `86d93cda…`、721 tensor・4,065,416,192-byte cacheとfull packは試験後に再照合し、共通target artifactも採用済み版へ戻した。現ソースの通常feature Wasmを新しくfull-graph検証した結果ではない。

生データは`artifacts/projection-reuse/{native,native-simd,probe,simd-probe}/report.json`、source hash、module、upload、primary-checkを同directoryへ保存しgitignore。再現は`check_projection_reuse.py --directory <fresh-output> --canister <専用ID> --wasm <candidate>`。部分packを別途sealed uploadした専用canisterに限り、スクリプトはinstallしない。SIMD版の不正状態検証は`--verify-malformed --native artifacts/projection-reuse/simd-primitive`を指定する。

## 追加の可逆通信試作

`explore_projection_state_codec.py`は、既に量子化済みの整数をそのまま1-byte INT8で保持し、F32 scale/A結果は4-byteの全bitを維持するPython試作である。7入力長で全stateのbit一致、範囲外整数・小数・NaN・truncation/checksumの拒否を確認した。132-tokenの状態frameは715,704 bytes、試作stateは377,040 bytes。ただし試作にはmodel/tensor識別の外側headerがなく、capture replyの出力との結合も未実装。Wasm codec・命令数・全通信・全層検証は未実施で、採用版の改善には計上しない。新たな中間状態の量子化ではなく、既存整数の可逆な表現変更を次の実装対象とする。

現在の全層採用版は354,704,377,597命令・305 query。初回prefix込み587 query、prefixなし485 queryで50/32回は未達。この候補はprefixなしの重複を主対象とし、prefix準備済み経路には同じ重複が0である。重複除去だけで目標を達成したとは扱わない。

後続で可逆INT8 codecとスケジューラを実装し、全5実行を検証した。現在のmodule/採用範囲・全層数値は[PROJECTION_CODEC.md](PROJECTION_CODEC.md)を参照。本文のF32配列経由の候補は採用していない。
