# Packed PCA32 Summary

- object size rows: 15
- steady-state rows: 4
- large-batch rows: 10
- throughput compare rows: 24

## Notes

- GL object serialization is unsupported in this build; object sizes use approximate Python object size only.
- CKKS object sizes use library serialize APIs.
- Dense CKKS large-batch runs may be skipped at high sample counts when runtime is prohibitive.
