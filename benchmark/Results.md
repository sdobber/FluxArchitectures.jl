# Results

## Julia 1.5.4 & Flux 0.11.5

### Info

```julia-repl
Julia Version 1.5.4
Commit 69fcb5745b (2021-03-11 19:13 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin18.7.0)
  CPU: Intel(R) Core(TM) i5-6267U CPU @ 2.90GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake)
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 4
```

```julia-repl
Status `~/Uni/Code/FluxArchitectures/benchmark/Project.toml`
  [fbb218c0] BSON v0.2.6
  [6e4b80f9] BenchmarkTools v0.5.0
  [a93c6f00] DataFrames v0.22.7
  [587475ba] Flux v0.11.5
  [5cadff95] JuliennedArrays v0.2.2
  [82cb661a] SliceMap v0.2.3
```

### Results

| Name     |  CPU        |   GPU     |   
|----------|-------------|-----------|
| DARNN    |     40.5469  |  missing |
| DSANet   |      3.2352  |  missing |
| LSTNet   |    0.905873  |  missing |
| TPA-LSTM |     9.29508  |  missing |


## Julia 1.6 & Flux 0.12.1

### Info

```julia-repl
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin19.6.0)
  CPU: Intel(R) Core(TM) i5-6267U CPU @ 2.90GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 4
```

```julia-repl
      Status `~/Uni/Code/FluxArchitectures/benchmark/Project.toml`
  [fbb218c0] BSON v0.3.3
  [6e4b80f9] BenchmarkTools v0.7.0
  [a93c6f00] DataFrames v0.22.7
  [587475ba] Flux v0.12.1
  [5cadff95] JuliennedArrays v0.2.2
  [82cb661a] SliceMap v0.2.4
  [e88e6eb3] Zygote v0.6.9
```

### Results

| Name     |  CPU        |  GPU     |   
|----------|-------------|----------|
| DARNN    |     11.579  |  missing |
| DSANet   |    3.05565  |  missing |
| LSTNet   |   0.683614  |  missing |
| TPA-LSTM |    7.71085  |  missing |