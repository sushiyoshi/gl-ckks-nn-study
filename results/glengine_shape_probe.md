# GLEngine Shape Probe

| ok | B | block | requested_shape | engine_shape | keygen_s | mm_ok | poly_ok | exception |
|---|---:|---:|---|---|---:|---|---|---|
| False | 1 | 16 | `[1, 16, 16]` | `None` |  | None | None | ValueError |
| False | 2 | 16 | `[2, 16, 16]` | `None` |  | None | None | ValueError |
| False | 4 | 16 | `[4, 16, 16]` | `None` |  | None | None | ValueError |
| False | 8 | 16 | `[8, 16, 16]` | `None` |  | None | None | ValueError |
| False | 16 | 16 | `[16, 16, 16]` | `None` |  | None | None | ValueError |
| False | 32 | 16 | `[32, 16, 16]` | `None` |  | None | None | ValueError |
| False | 64 | 16 | `[64, 16, 16]` | `None` |  | None | None | ValueError |
| False | 128 | 16 | `[128, 16, 16]` | `None` |  | None | None | ValueError |
| False | 256 | 16 | `[256, 16, 16]` | `[256, 16, 16]` | 0.333 | True | None | RuntimeError |
| False | 1 | 32 | `[1, 32, 32]` | `None` |  | None | None | ValueError |
| False | 2 | 32 | `[2, 32, 32]` | `None` |  | None | None | ValueError |
| False | 4 | 32 | `[4, 32, 32]` | `None` |  | None | None | ValueError |
| False | 8 | 32 | `[8, 32, 32]` | `None` |  | None | None | ValueError |
| False | 16 | 32 | `[16, 32, 32]` | `None` |  | None | None | ValueError |
| False | 32 | 32 | `[32, 32, 32]` | `None` |  | None | None | ValueError |
| False | 64 | 32 | `[64, 32, 32]` | `None` |  | None | None | ValueError |
| False | 128 | 32 | `[128, 32, 32]` | `None` |  | None | None | ValueError |
| True | 256 | 32 | `[256, 32, 32]` | `[256, 32, 32]` | 6.662 | True | True |  |
| False | 1 | 64 | `[1, 64, 64]` | `None` |  | None | None | ValueError |
| False | 2 | 64 | `[2, 64, 64]` | `None` |  | None | None | ValueError |
| False | 4 | 64 | `[4, 64, 64]` | `None` |  | None | None | ValueError |
| False | 8 | 64 | `[8, 64, 64]` | `None` |  | None | None | ValueError |
| False | 16 | 64 | `[16, 64, 64]` | `None` |  | None | None | ValueError |
| False | 32 | 64 | `[32, 64, 64]` | `None` |  | None | None | ValueError |
| False | 64 | 64 | `[64, 64, 64]` | `None` |  | None | None | ValueError |
| False | 128 | 64 | `[128, 64, 64]` | `None` |  | None | None | ValueError |
| False | 256 | 64 | `[256, 64, 64]` | `None` |  | None | None | TimeoutExpired |
