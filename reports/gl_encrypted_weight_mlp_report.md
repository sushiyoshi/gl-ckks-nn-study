# desilofhe GL encrypted-weight MLP / Transformer FFN 実験レポート

## 1. 概要

本レポートは、M2 Mac 上で実施した desilofhe の GL encrypted-weight MLP 実験結果をまとめる。主対象は、入力と重みを GL ciphertext として扱う encrypted-weight MLP であり、PCA32 特徴量に対する小規模 MLP と、Transformer FFN 相当の 512→2048→512 コンポーネントを評価した。

まず、32→32→10 の小規模 PCA32 MLP で、8192 samples full packing が通った。8192 サンプルを 1.0 の packing utilization で処理し、暗号実行結果は平文参照に対して argmax agreement 1.0、relative L2 error 7.05e-08 を示した。

さらに、今回の追加実験では、synthetic な Transformer FFN コンポーネント 512→2048→512 を GL encrypted-weight と plaintext-weight の両方で実際に実行した。`shape=(16,512,512)`、`block_size=512` を用いることで 8192 samples を 1 ciphertext pack に収め、8 block matmuls と 4 activation blocks で計算できた。encrypted-weight の実行結果は allclose=True、relative L2 error 7.65e-06、plaintext-weight の実行結果は allclose=True、relative L2 error 3.49e-07 で、どちらも semantic validation を通過した。

Transformer FFN では、重みを ciphertext として扱う ct-ct 乗算版は plaintext-weight 版に対して server-only で約 1.58x、total で約 1.57x 遅かった。重みプライバシーの追加コストは明確だが、実行時間の絶対値は plaintext-weight でも大きく、本結果は packed 評価の成立確認と、GL の大きな supported shape を使った Transformer FFN コンポーネント実行の初期実測として位置づける。

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

評価対象は、PCA32 特徴量を入力とする encrypted-weight MLP と、Transformer FFN 相当の synthetic MLP コンポーネントである。

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

追加評価した Transformer FFN コンポーネントは以下である。

| 項目 | 値 |
| --- | --- |
| encrypted-weight result file | `results/gl_transformer_ffn_512_2048_512_shape_16_512_512_run.json` |
| plaintext-weight result file | `results/gl_transformer_ffn_512_2048_512_shape_16_512_512_plain_weight_run.json` |
| task_type | `transformer_ffn_component_poly_relu` |
| activation | `degree3_poly` |
| polynomial_degree | 3 |
| polynomial_radius | 3.0 |
| logical_dims | `[512, 2048, 512]` |
| n_samples | 8192 |
| selected_shape | `[16, 512, 512]` |
| block_size | 512 |
| packing utilization | 1.0 |

これは Transformer 全体ではなく、FFN 部分のみの synthetic component 実験である。attention、softmax、layernorm、bootstrapping は含まない。

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

## 9. Transformer FFNコンポーネント実測

今回追加した実測では、`scripts/63_run_gl_transformer_ffn_supported_shape.py` と `scripts/65_run_gl_transformer_ffn_plain_weight.py` により 512→2048→512 の FFN コンポーネントを `shape=(16,512,512)` で明示実行した。encrypted-weight 版は入力と重みを GL ciphertext とする ct-ct 乗算であり、plaintext-weight 版は入力を ciphertext、重みを plaintext encode とする。どちらも bias は plaintext encode、活性化は degree-3 polynomial ReLU approximation とした。

encrypted-weight 版の主要結果は以下の通り。

| 指標 | 値 |
| --- | ---: |
| semantic_validation_passed | true |
| allclose | true |
| logits_linf | 9.99790878647433e-06 |
| logits_mae | 8.585758561423912e-07 |
| logits_relative_l2 | 7.646037265198966e-06 |
| key_generation_s | 119.16942658299376 |
| server_only_s | 2015.8906234989845 |
| total_s | 2216.0641234170034 |
| total_minus_keygen_s | 2096.8946968340097 |
| server_per_sample_ms | 246.08039837634088 |
| no_keygen_per_sample_ms | 255.9685909221203 |
| total_per_sample_ms | 270.5156400655522 |

8192 samples full packing で total は約 36.93 分、server-only は約 33.60 分だった。per-sample 値は 8192 samples 一括処理時の平均コストであり、単一サンプルの低レイテンシ実行として解釈すべきではない。

plaintext-weight 版との比較は以下の通り。

| 指標 | plaintext-weight | encrypted-weight ct-ct | 比率 |
| --- | ---: | ---: | ---: |
| ciphertext_count | 1 | 9 | 9.00x |
| weight_ciphertext_count | 0 | 8 | - |
| server_only_s | 1272.3728133330078 | 2015.8906234989845 | 1.58x |
| total_s | 1411.590886124999 | 2216.0641234170034 | 1.57x |
| total_minus_keygen_s | 1297.875530958001 | 2096.8946968340097 | 1.62x |
| server_per_sample_ms | 155.31894694006443 | 246.08039837634088 | 1.58x |
| relative_l2 | 3.4946763240689663e-07 | 7.646037265198966e-06 | 21.88x |
| linf | 3.997482766471272e-07 | 9.99790878647433e-06 | 25.01x |
| mae | 4.368988036080161e-08 | 8.585758561423912e-07 | 19.65x |
| input encryption s | 6.960596666001948 | 7.541467666000244 | 1.08x |
| weight encode/encryption s | 5.150920749998477 | 57.71324920900224 | 11.20x |

両者は同じ `dims=[512,2048,512]`、同じ `shape=[16,512,512]`、同じ 8 block matmuls で実行されている。そのため、この比較は主に「重みを plaintext として扱うか、ciphertext として扱うか」の差を表している。ct-ct 版では重み 8 blocks が ciphertext になり、重み暗号化時間と matrix multiply の実行時間が増える。精度面でも誤差は約 20〜25 倍大きくなるが、今回の条件では allclose=True の範囲に収まった。

スケジュールは以下の通りである。

| 項目 | 値 |
| --- | ---: |
| input_blocks | 1 |
| hidden_blocks | 4 |
| output_blocks | 1 |
| linear1_block_matmuls | 4 |
| linear2_block_matmuls | 4 |
| total_linear_block_matmuls | 8 |
| activation_blocks | 4 |
| input_ciphertext_count | 1 |
| weight_ciphertext_count | 8 |
| peak_ciphertext_count_estimate | 14 |

`shape=(16,512,512)` は sample capacity が 8192 であり、今回の n_samples と一致する。そのため sample packing utilization、input/hidden/output の feature padding utilization はすべて 1.0 だった。

timing breakdown は以下の通り。

| 処理 | 時間 s | server-only比 |
| --- | ---: | ---: |
| linear1 total | 1361.6585184579963 | 67.5% |
| activation total | 261.0467013739908 | 12.9% |
| linear2 total | 393.18540366699744 | 19.5% |

最も重いのは 1 層目の 512→2048 projection である。4 つの linear1 block matmul はそれぞれ約 339〜342 秒で、server-only 時間の約 3 分の 2 を占めた。linear2 は 4 block 合計で約 393 秒、activation は 4 block 合計で約 261 秒だった。

level log は、入力と重みが level 6、linear1 後が level 5、activation 後が level 2、linear2 後が level 1 である。小規模 MLP と同様に、degree-3 polynomial activation が level を大きく消費する。今回の 1 hidden layer FFN コンポーネントでは最終出力まで level が残ったが、同じ構成を多層に積む場合は bootstrapping なしでは level budget がすぐ制約になる。

## 10. 実装上の更新点

今回の実装では、既存の generic encrypted-weight MLP 経路を Transformer FFN コンポーネントにも使えるようにした。

- `src/gl_block_schedule.py` は、shape と block_size に基づいて input/hidden/output blocks、linear block matmul 数、activation block 数、logical ciphertext count を計算する。
- `src/gl_shape_selector.py` は、desilofhe GLEngine の supported shape 候補を扱い、`(16,512,512)` のような大きな square block shape を明示選択できる。
- `src/gl_block_linear.py` は、特徴量を block に分割し、サンプルを GL tensor の column 方向へ pack し、重み block を batch 軸へ broadcast する。
- `src/gl_generic_mlp.py` は、平文 GL-layout validation、暗号化 input/weight、block linear、polynomial activation、復号、誤差評価、level/timing log を共通化している。
- `scripts/63_run_gl_transformer_ffn_supported_shape.py` は、synthetic 512→2048→512 FFN を encrypted-weight で生成し、supported shape 選択、dry-run、actual run の JSON/CSV 出力を行う。
- `scripts/65_run_gl_transformer_ffn_plain_weight.py` は、同じ synthetic FFN を plaintext-weight で実行し、ct-ct 版との差分を比較できる JSON/CSV 出力を行う。

重要な点は、512 次元を `block_size=512` の 1 input block として扱い、2048 hidden を 4 hidden blocks、512 output を 1 output block として扱ったことである。これにより、従来の `block_size=32` なら 1280 block matmuls になりうる 512→2048→512 の FFN を、8 block matmuls まで落とせた。ただし、1 回の 512×512 GL matrix multiply は重く、実測では linear1 の 1 block matmul あたり約 340 秒かかった。

## 11. 考察

今回の実験で確認できたことは、32→32→10 の小規模 PCA32 MLP に対して、入力と重みを暗号化した GL encrypted-weight 構成で 8192 samples full packing が成立した点である。分類結果は平文参照と一致し、relative L2 error も小さい。

加えて、512→2048→512 の Transformer FFN 相当コンポーネントでも、supported shape を適切に選べば GL encrypted-weight と plaintext-weight の両方が完走することを確認した。`shape=(16,512,512)` は 8192 samples をちょうど収め、feature padding も発生しないため、今回の synthetic FFN には都合がよい。

性能面では、full packing によって per-sample の平均時間は下がっている。ただし、平文 baseline と比べると server-only でも数千倍の slowdown がある。Transformer FFN では、ct-ct 版は plaintext-weight 版より server-only で約 743.5 秒遅く、total で約 804.5 秒遅い。これは重みプライバシーの追加コストとして測れるが、plaintext-weight 版自体も total 約 23.53 分を要するため、512×512 block matmul の絶対コストが支配的である。

この結果は、packed batch 処理における小規模 MLP の成立確認としては有用である。Transformer FFN コンポーネントについても、単一 FFN ブロックが指定条件で通ることを示した。一方で、Transformer 全体の実用性を示すものではない。attention、softmax、layernorm、residual、複数層、bootstrapping を含めた評価はまだ行っていない。

## 12. 結論

M2 Mac 上の desilofhe 環境で、32→32→10 の小規模 PCA32 MLP に対し、8192 samples full packing の encrypted-weight GL 実行が成功した。また、synthetic な 512→2048→512 Transformer FFN コンポーネントについても、`shape=(16,512,512)` を用いた encrypted-weight と plaintext-weight の actual run が成功した。

主要な確認結果は以下である。

- accuracy は 0.9454345703125。
- argmax_agreement は 1.0。
- relative_l2 は 7.051293925400409e-08。
- server_only_s は 9.750431624994235 秒。
- total_s は 18.13020904199948 秒。
- packing utilization は 1.0。
- 最終 linear2 後の level は 1。

Transformer FFN コンポーネントの主要な確認結果は以下である。

- dims は 512→2048→512。
- selected_shape は `[16, 512, 512]`。
- n_samples は 8192。
- total block matmuls は 8。
- activation blocks は 4。
- logits_relative_l2 は 7.646037265198966e-06。
- allclose は true。
- server_only_s は 2015.8906234989845 秒。
- total_s は 2216.0641234170034 秒。
- plaintext-weight の server_only_s は 1272.3728133330078 秒。
- plaintext-weight の total_s は 1411.590886124999 秒。
- ct-ct 版は plaintext-weight 版より server-only で約 1.58x 遅い。
- ct-ct 版は plaintext-weight 版より relative_l2 が約 21.88x 大きい。
- 最終 linear2 後の level は 1。

以上から、今回の条件では小規模 PCA32 MLP と単一 Transformer FFN コンポーネントの packed encrypted-weight 評価は成立した。さらに Transformer FFN では、plaintext-weight と ct-ct の差分を同一 shape、同一 n_samples で測定できた。ただし、平文実行に対する slowdown は大きく、実用上の評価にはモデル規模、batch 条件、level budget、鍵生成や暗号化を含めるかどうかを分けた追加検証が必要である。

## 13. 今後の課題

今後の課題は以下である。

- n_samples を変えたときの packing utilization と per-sample cost の関係を整理する。
- hidden dimension、層数、出力次元を変えた場合の level 消費と実行時間を測る。
- Transformer FFN について、`shape=(16,512,512)` 以外の supported shape や batch 条件での性能を比較する。
- Transformer FFN の plaintext-weight と encrypted-weight ct-ct の差分を、hidden dimension や block_size を変えて測定する。
- attention、softmax、layernorm、residual connection を含む Transformer 全体への拡張可能性を分解して評価する。
- polynomial activation の次数や radius を変え、精度、level 消費、実行時間の trade-off を確認する。
- key generation、encryption、server-only、decryption を分けた測定を継続し、用途別にどの時間を見るべきかを明確にする。
- OpenFHE-NumPy baseline と比較する場合は、環境、パラメータ、packing 条件、測定範囲をそろえて別レポートとして扱う。
