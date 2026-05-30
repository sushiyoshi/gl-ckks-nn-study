# GL Block Matmul Count Estimates

Metadata: estimates only; not a performance benchmark. Block size is 32.

| Model | Type | Dims | Total block matmuls |
|---|---:|---:|---:|
| PCA32 toy | MLP | 32->32->10 | 2 |
| raw64 digits | MLP | 64->32->10 | 3 |
| original Transformer base FFN | MLP | 512->2048->512 | 2048 |
| GPT-3 175B rough FFN | MLP | 12288->49152->12288 | 1179648 |
| Llama2-7B style SwiGLU | SwiGLU | hidden=4096, intermediate=11008 | 132096 |
| Llama3-8B style SwiGLU | SwiGLU | hidden=4096, intermediate=14336 | 172032 |
