# desilofhe GL encrypted-weight MLP 実験レポート

## 1. 概要

本レポートは、M2 Mac 上で実施した desilofhe の GL encrypted-weight MLP 実験結果をまとめる。主対象は、入力と重みを GL ciphertext として扱う encrypted-weight MLP であり、PCA32 特徴量に対する小規模 MLP を評価した。

今回確認できた主な結果は、32→32→10 の小規模 PCA32 MLP で、8192 samples full packing が通ったことである。8192 サンプルを 1.0 の packing utilization で処理し、暗号実行結果は平文参照に対して argmax agreement 1.0、relative L2 error 7.05e-08 を示した。

一方で、平文 baseline と比較した実行時間差は大きい。暗号化によるオーバーヘッドは依然として支配的であり、本結果は小規模 MLP に対する packed encrypted-weight 評価の成立確認として位置づける。

## 2. 実行環境

実行環境は以下の通り。

| 項目 | 値 |
| --- | --- |
| Hardware | MacBook Pro M2 Pro |
| OS | macOS 15.3.1 |
| Python | 3.13.4 |
| desilofhe | import OK |
| numpy | 2.4.6 |
| sklearn | 1.8.0 |
| git revision | `aa2e95b841980842aa8df72906808d57b7803b01` |

Ubuntu/OpenFHE-NumPy 環境については、本レポートでは深掘りしない。

## 3. 評価対象

評価対象は、PCA32 特徴量を入力とする encrypted-weight MLP である。

| 項目 | 値 |
| --- | --- |
| result file | `results/gl_encrypted_weight_pca32_mlp_n8192.json` |
| task_type | `mlp_poly_relu` |
| activation | `degree3_poly` |
| polynomial_degree | 3 |
| polynomial_radius | 6.0 |
| logical_dims | `[32, 32, 10]` |
| n_samples | 8192 |
| packing utilization | 1.0 |

モデル構造は 32→32→10 の MLP で、ReLU は degree-3 polynomial approximation に置き換えている。実験は GL layout 上で実行され、8192 サンプルが packing capacity を使い切る形で評価された。

## 4. 暗号化MLPの結果

暗号化 MLP の主要結果は以下の通り。

| 指標 | 値 |
| --- | ---: |
| accuracy | 0.9454345703125 |
| argmax_agreement | 1.0 |
| relative_l2 | 7.051293925400409e-08 |
| key_generation_s | 6.289987874999497 |
| server_only_s | 9.750431624994235 |
| total_s | 18.13020904199948 |
| total_minus_keygen_s | 11.840221166999982 |
| server_per_sample_ms | 1.190238235472929 |
| no_keygen_per_sample_ms | 1.4453394979248024 |
| total_per_sample_ms | 2.213160283447202 |

argmax agreement が 1.0 であるため、今回の 8192 サンプルでは暗号実行後の分類結果が平文参照と一致した。relative L2 error も 7.05e-08 と小さく、数値的な差は分類結果に影響していない。

timing breakdown は以下の通り。

| 処理 | 時間 s |
| --- | ---: |
| encryption_input_s | 0.472336 |
| encryption_weight_s | 0.941036 |
| linear1_matrix_multiply_s | 4.553193 |
| activation_s | 3.721800 |
| linear2_matrix_multiply_s | 1.475439 |
| decryption_s | 0.469095 |

server-only 時間の中では、1 層目の matrix multiply と polynomial activation が大きな割合を占めている。

## 5. level消費

level log は以下の通り。

| stage | level |
| --- | ---: |
| input | 6 |
| weight_w1 | 6 |
| weight_w2 | 6 |
| linear1 | 5 |
| activation | 2 |
| linear2 | 1 |

入力と重みは level 6 から開始し、1 回目の線形層で level 5、degree-3 polynomial activation 後に level 2、2 回目の線形層後に level 1 となった。今回の構成では最終出力まで level が残っており、32→32→10 の 1 hidden layer MLP は評価可能だった。

ただし、activation で level を大きく消費しているため、同じ方針で層数を増やす場合は level budget が主要な制約になる。

## 6. packingによるamortization

今回の encrypted MLP は、8192 サンプルを full packing で処理している。

| 項目 | 値 |
| --- | ---: |
| n_samples | 8192 |
| samples_per_ciphertext_capacity | 8192 |
| sample_packing_utilization | 1.0 |
| used_batches | 256 |
| used_columns_last_batch | 32 |
| feature_dim | 32 |

server_only_s は 9.750431624994235 秒だが、8192 サンプルで amortize すると server_per_sample_ms は 1.190238235472929 ms になる。total では key generation、input encryption、weight encryption、decryption を含むため total_per_sample_ms は 2.213160283447202 ms である。

この per-sample 値は full packing による amortization の効果を含む。少数サンプルのレイテンシとして解釈するのではなく、8192 サンプル一括処理時の平均コストとして扱うべきである。

## 7. 平文baselineとの比較

平文 baseline は `results/plaintext_vs_gl_encrypted_mlp_n8192.json` に記録されている。

| 項目 | 値 |
| --- | --- |
| model | `data/pca32_mlp.joblib` |
| weights | actual PCA32 model / W1 / W2 / bias |
| degree3 coefficients | PCA32 train z1 から生成 |
| radius | 6.0 |
| GL-layout validation | allclose=True |

平文実行時間の中央値は以下の通り。

| baseline | median s |
| --- | ---: |
| plaintext logical MLP | 0.0015505829996982357 |
| plaintext GL-layout MLP | 0.004537478998827282 |

暗号化 MLP との slowdown は以下の通り。

| 比較 | slowdown |
| --- | ---: |
| encrypted server-only vs plaintext logical | 6288.235861538402x |
| encrypted no-keygen vs plaintext logical | 7635.980253429999x |
| encrypted server-only vs plaintext GL-layout | 2148.865400261742x |
| encrypted no-keygen vs plaintext GL-layout | 2609.42721058546x |

GL-layout validation は allclose=True であり、平文 logical MLP と平文 GL-layout MLP は同じ計算結果を返している。そのため、暗号化結果との差分は layout 実装差ではなく、暗号実行に伴う数値誤差と実行コストとして評価できる。

## 8. OpenFHE-NumPy baselineについて

OpenFHE-NumPy baseline については、今回の主対象ではない。Ubuntu/OpenFHE-NumPy 環境の詳細や比較条件の掘り下げは本レポートの範囲外とする。

本レポートでは、M2 Mac 上の desilofhe GL encrypted-weight MLP 実験を中心に扱う。したがって、OpenFHE-NumPy との厳密な性能比較や環境差の分析は行わない。

## 9. 考察

今回の実験で確認できたことは、32→32→10 の小規模 PCA32 MLP に対して、入力と重みを暗号化した GL encrypted-weight 構成で 8192 samples full packing が成立した点である。分類結果は平文参照と一致し、relative L2 error も小さい。

性能面では、full packing によって per-sample の平均時間は下がっている。ただし、平文 baseline と比べると server-only でも数千倍の slowdown がある。特に matrix multiply と polynomial activation が主要な実行時間要因であり、activation は level 消費の面でも重い。

この結果は、packed batch 処理における小規模 MLP の成立確認としては有用である。一方で、より大きなネットワーク、深い MLP、あるいは Transformer 級モデルの実用性を示すものではない。今回の結論は、あくまで PCA32 の 1 hidden layer MLP が指定条件で通った、という範囲に限定する。

## 10. 結論

M2 Mac 上の desilofhe 環境で、32→32→10 の小規模 PCA32 MLP に対し、8192 samples full packing の encrypted-weight GL 実行が成功した。

主要な確認結果は以下である。

- accuracy は 0.9454345703125。
- argmax_agreement は 1.0。
- relative_l2 は 7.051293925400409e-08。
- server_only_s は 9.750431624994235 秒。
- total_s は 18.13020904199948 秒。
- packing utilization は 1.0。
- 最終 linear2 後の level は 1。

以上から、今回の条件では小規模 PCA32 MLP の packed encrypted-weight 評価は成立した。ただし、平文実行に対する slowdown は大きく、実用上の評価にはモデル規模、batch 条件、level budget、鍵生成や暗号化を含めるかどうかを分けた追加検証が必要である。

## 11. 今後の課題

今後の課題は以下である。

- n_samples を変えたときの packing utilization と per-sample cost の関係を整理する。
- hidden dimension、層数、出力次元を変えた場合の level 消費と実行時間を測る。
- polynomial activation の次数や radius を変え、精度、level 消費、実行時間の trade-off を確認する。
- key generation、encryption、server-only、decryption を分けた測定を継続し、用途別にどの時間を見るべきかを明確にする。
- OpenFHE-NumPy baseline と比較する場合は、環境、パラメータ、packing 条件、測定範囲をそろえて別レポートとして扱う。
