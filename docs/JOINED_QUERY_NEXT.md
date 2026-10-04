# 連結スケジュールの次の探索

MLP完了→次Attentionを一回にする経路は、全体へ接続してからquery数・全保持state・判断を確認する。部分経路の成功だけで全体の50/32到達を主張しない。

現在の候補は8層を13query（入口3＋五連結5＋後半5）で進める。0〜23層39query、24〜28層8query、29層とDelta30の3query、既存のterminal tail1queryという**51queryを狙う構成**。実測は後続文書へ記録する。132総tokenを切り詰めず、主suffix87、prefix45、rotations=1を維持する。

最後の1queryを減らす候補は二方向。まず終端Deltaの残りheadをtailへ融合し、5B以内か実測する。ただし残り12headの計算を4.7B前後のtailへ追加すると超過する可能性が高い。単に連結して成功を仮定しない。

別案は最後のブロックでQの配分とMLP前半の配分を変える。Delta26の入力量子化・F32 A・gates captureだけを直前queryへ移し、Delta26 FULLとMLP26前半をより多く進める。次のqueryでAttention27のQ12を処理し、続くQ4のqueryでMLP27 gate/upを全て終える。MLP27 downだけ＋Delta28 FULL、MLP28 FULL＋Delta29の先頭、Delta29残り＋MLP29前半、MLP29完了＋Delta30 FULL、既存tailへ接続する候補である。

この案には、完成したMLP carryから不要なinput q/gate-up Aを省く専用型、Q12 carryから後続不要なMLP normを省くcodec、prepared Delta captureを再使用するAPIが必要。H完了のproduct INT8/scale/down F32 A/残差BF16を保持し、既存pipelineのcarryとビット比較する。prefixの可逆Huffmanを使ってもframeが2MB以内か個別に確認する。配分を変えるだけで総命令が減るとは限らず、通信・handler命令・時間を別々に報告する。

上記は未実装・未検証の探索案。新経路は通常query・client保持の中間状態を用い、固定モデルの準備だけupdateを使う。採用INT8版へのbit一致と、元BF16モデルに対する判断精度は別の検証を維持する。
