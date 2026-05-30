# GL plain-weight inference unsupported

Reason: GLEngine.matrix_multiply accepts plaintext weights, but this padded vector-to-GL-matrix mapping did not reproduce the plaintext polynomial MLP within tolerance.

Details:

```json
{
  "n_samples_requested": 1,
  "shape": [
    256,
    32,
    32
  ],
  "slot_count": 262144,
  "packing_utilization_input": 0.000247955322265625,
  "padding_input": 262079,
  "key_generation_s": 6.425865249999333,
  "plain_weight_matrix_multiply": {
    "numpy_16x64_times_ct": {
      "ok": true,
      "type": "GLCiphertext",
      "level": 5,
      "nbytes": null,
      "serialized_nbytes": null
    },
    "numpy_full_shape_times_ct": {
      "ok": true,
      "type": "GLCiphertext",
      "level": 5,
      "nbytes": null,
      "serialized_nbytes": null
    },
    "glplaintext_full_shape_times_ct": {
      "ok": true,
      "type": "GLCiphertext",
      "level": 5,
      "nbytes": null,
      "serialized_nbytes": null
    }
  },
  "accuracy_fhe_decrypted": 0.0,
  "argmax_agreement": 0.0,
  "fhe_numeric_logits_linf": 5.051333712547087,
  "fhe_numeric_logits_mae": 2.4697806072616553,
  "fhe_numeric_logits_relative_l2": 1.0532209523202583,
  "semantic_validation_supported": false
}
```
