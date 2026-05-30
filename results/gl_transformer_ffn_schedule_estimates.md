# GL Transformer FFN block schedule estimates

These are simple linear extrapolations from observed per-block times. They do not include memory, cache, level, bootstrapping, scheduling, or key/material pressure effects.

| model | kind | dims | input_blocks | hidden_blocks | intermediate_blocks | output_blocks | total_block_matmuls | activation_blocks | estimated_server_only_s_raw64_block_time | estimated_server_only_s_pca32_block_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCA32 toy | FFN | 32->32->10 | 1 | 1 |  | 1 | 2 | 1 | 9.574138056668744 | 9.750431624994235 |
| raw64 digits | FFN | 64->32->10 | 2 | 1 |  | 1 | 3 | 1 | 14.361207085003116 | 14.625647437491352 |
| synthetic medium | FFN | 256->512->256 | 8 | 16 |  | 8 | 256 | 16 | 1225.4896712535992 | 1248.055247999262 |
| original Transformer FFN | FFN | 512->2048->512 | 16 | 64 |  | 16 | 2048 | 64 | 9803.917370028794 | 9984.441983994097 |
| Llama2-7B style SwiGLU | SwiGLU | hidden=4096, intermediate=11008 | 128 |  | 344 | 128 | 132096 | 344 | 632352.6703668572 | 643996.5079676192 |
| Llama3-8B style SwiGLU | SwiGLU | hidden=4096, intermediate=14336 | 128 |  | 448 | 128 | 172032 | 448 | 823529.0590824187 | 838693.1266555041 |
