# Report input summary

## Key JSON result summary

| file | task_type | n_samples | activation | sample_packing_utilization | server_only_s | total_s | total_minus_keygen_s | server_per_sample_s | no_keygen_per_sample_s | semantic_validation_passed | relative_l2 | accuracy | argmax_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| results/gl_ctct_matrix_multiply_probe.json | None | None | None | None | None | 18.2369 | 11.7361 | None | None | True | None | None | None |
| results/gl_encrypted_weight_mutation_tests.json | None | 32 | None | None | None | None | None | None | None | None | None | None | None |
| results/gl_encrypted_weight_pca32_linear_n1.json | linear | 1 | none | 0.00012207 | None | 13.0157 | 6.51669 | None | 6.51669 | True | 8.2713e-08 | None | None |
| results/gl_encrypted_weight_pca32_linear_n32.json | linear | 32 | none | 0.00390625 | None | 12.9904 | 6.51156 | None | 0.203486 | True | 5.49306e-08 | None | None |
| results/gl_encrypted_weight_pca32_linear_n450.json | linear | 450 | none | 0.0549316 | 4.5026 | 12.5948 | 6.28709 | 0.0100058 | 0.0139713 | True | 6.00677e-08 | None | None |
| results/gl_encrypted_weight_pca32_linear_n8192.json | linear | 8192 | none | 1 | 4.63659 | 12.9227 | 6.47642 | 0.00056599 | 0.000790579 | True | 5.91568e-08 | None | None |
| results/gl_encrypted_weight_pca32_mlp_n1.json | mlp_poly_relu | 1 | degree3_poly | 0.00012207 | None | 18.7008 | 12.2281 | None | 12.2281 | True | 4.6107e-08 | 1 | 1 |
| results/gl_encrypted_weight_pca32_mlp_n32.json | mlp_poly_relu | 32 | degree3_poly | 0.00390625 | None | 18.6344 | 12.1265 | None | 0.378952 | True | 5.08475e-08 | 0.9375 | 1 |
| results/gl_encrypted_weight_pca32_mlp_n450.json | mlp_poly_relu | 450 | degree3_poly | 0.0549316 | 10.1247 | 18.7557 | 12.3638 | 0.0224994 | 0.0274751 | True | 6.91741e-08 | 0.948889 | 1 |
| results/gl_encrypted_weight_pca32_mlp_n8192.json | mlp_poly_relu | 8192 | degree3_poly | 1 | 9.75043 | 18.1302 | 11.8402 | 0.00119024 | 0.00144534 | True | 7.05129e-08 | 0.945435 | 1 |
| results/gl_encrypted_weight_pca32_two_linear_n1.json | two_linear | 1 | none | 0.00012207 | None | 17.0772 | 10.7133 | None | 10.7133 | True | 5.46738e-08 | 0 | 1 |
| results/gl_encrypted_weight_pca32_two_linear_n32.json | two_linear | 32 | none | 0.00390625 | None | 17.059 | 10.7639 | None | 0.336372 | True | 7.35391e-08 | 0.875 | 1 |
| results/gl_encrypted_weight_pca32_two_linear_n450.json | two_linear | 450 | none | 0.0549316 | 8.55337 | 17.3553 | 10.9307 | 0.0190075 | 0.0242905 | True | 6.21706e-08 | 0.893333 | 1 |
| results/openfhe_numpy_ckks_matmul_sweep.json | None | None | None | None | None | None | None | None | None | True | None | None | None |
| results/openfhe_numpy_ckks_matmul_toy.json | None | None | None | None | None | 4.58e-07 | None | None | None | False | None | None | None |
| results/openfhe_numpy_env_probe.json | None | None | None | None | None | None | None | None | None | None | None | None | None |
| results/pca32_train.json | None | None | None | None | None | None | None | None | None | None | None | None | None |
