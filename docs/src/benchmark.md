# Benchmarks

FluxArchitectures is included in the [FluxBench](https://github.com/FluxML/FluxBench.jl)-suite, with results available at [speed.fluxml.ai](https://speed.fluxml.ai/). 

To run the benchmarks yourself, add FluxBench as a package and type
```julia
using FluxBench
FluxBench.submit()
```

The following sections give the timings for my own hardware (which is very limited as compared to current state-of-the-art equipment).

!!! note
    The setup does not ensure consistency in the number of parameters etc. between models - comparisons are only useful between GPU and CPU runs of the same model, and runs for different versions of FluxArchitectures. In addition, as the benchmark only performs a single backward pass, differences between CPU and GPU are not necessarily indicative of actual training performance. 

Currently, only `DSANet` sees an improvement in training speed on a GPU, whereas all other models train faster on a CPU. The deeper reason is the current limitation of Flux's implementation of RNNs, see [this issue](https://github.com/FluxML/Flux.jl/issues/1365) that affects all models except `DSANet`, which doesn't use recurrence. 


## JetsonNano

The benchmark is run on a [JetsonNano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) with 4GB of memory.

|          | Forward Pass |           | Backward Pass |            |
|              | CPU       | GPU           | CPU        | GPU      |
| ------------ | --------- | ------------- | ---------- | -------- |
| DARNN        | 91.290 ms | 13.808 s      | 1.810 s    | 39.654 s |
| DSAnet       | 222.675ms | 556.588 ms    | 796.776 ms | 1.409 s  |
| LSTNet       | 18.256 ms | 1.452 s       | 428.595 ms | 7.799 s  |
| TPA-LSTM     | 23.069 ms | 5.632 s       | 1.572 s    | 18.750 s |


## Intel Core i5 (2.9 GHz)

The benchmark is run on a [13" MacBook Pro (Late 2016)](https://support.apple.com/kb/SP748?viewlocale=en_US&locale=da_DK) without GPU support.

|              | Forward Pass  | Backward Pass|
|              | CPU           | CPU         |
| ------------ | ------------- | ----------- |
| DARNN        | 29.191 ms    | 528.621 ms |
| DSAnet       | 137.335 ms   | 395.298 ms |
| LSTNet       | 3.455 ms     | 103.729 ms |
| TPA-LSTM     | 7.885 ms     | 435.726 ms |


## Technical Details

Version information for Julia and CUDA for the different devices for reference.


### JetsonNano

```
julia> versioninfo(verbose=true)
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: Linux (aarch64-unknown-linux-gnu)
      Ubuntu 18.04.6 LTS
  uname: Linux 4.9.140-tegra #1 SMP PREEMPT Tue Oct 27 21:02:37 PDT 2020 aarch64 aarch64
  CPU: unknown: 
              speed         user         nice          sys         idle          irq
       #1   710 MHz      12735 s          0 s        562 s      12506 s        150 s
       #2   518 MHz       2904 s          0 s        337 s      22803 s         31 s
       #3  1479 MHz       2901 s          0 s        382 s      22706 s         35 s
       #4  1479 MHz       2701 s          0 s        350 s      22992 s         22 s
       
  Memory: 3.8712120056152344 GB (178.02734375 MB free)
  Uptime: 2629.0 sec
  Load Avg:  0.48  0.95  0.97
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cortex-a57)
Environment:
  HOME = /home/sdobber
  TERM = xterm-256color
  PATH = /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/sdobber/Programme/julia-1.6.2/bin
```

```
julia> FluxBench.Flux.CUDA.versioninfo()
CUDA toolkit 10.2.89, local installation
CUDA driver 10.2.0

Libraries: 
- CUBLAS: 10.2.2
- CURAND: 10.1.2
- CUFFT: 10.1.2
- CUSOLVER: 10.3.0
- CUSPARSE: 10.3.1
- CUPTI: 12.0.0
- NVML: missing
- CUDNN: 8.0.0 (for CUDA 10.2.0)
- CUTENSOR: missing

Toolchain:
- Julia: 1.6.2
- LLVM: 11.0.1
- PTX ISA support: 3.2, 4.0, 4.1, 4.2, 4.3, 5.0, 6.0, 6.1, 6.3, 6.4, 6.5
- Device capability support: sm_30, sm_32, sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75

1 device:
  0: NVIDIA Tegra X1 (sm_53, 66.707 MiB / 3.871 GiB available)
```


## Intel Core i5 (2.9 GHz )

```
julia> versioninfo(verbose=true)
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin18.7.0)
  uname: Darwin 18.7.0 Darwin Kernel Version 18.7.0: Tue Jun 22 19:37:08 PDT 2021; root:xnu-4903.278.70~1/RELEASE_X86_64 x86_64 i386
  CPU: Intel(R) Core(TM) i5-6267U CPU @ 2.90GHz: 
              speed         user         nice          sys         idle          irq
       #1  2900 MHz     206462 s          0 s      94219 s     732080 s          0 s
       #2  2900 MHz      70845 s          0 s      29756 s     931955 s          0 s
       #3  2900 MHz     210310 s          0 s      72773 s     749481 s          0 s
       #4  2900 MHz      60839 s          0 s      25704 s     946013 s          0 s
       
  Memory: 16.0 GB (5212.79296875 MB free)
  Uptime: 1.118083e6 sec
  Load Avg:  1.67041015625  1.7412109375  1.875
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 4
  HOME = /Users/sdobber
  PATH = /usr/local/lib/ruby/gems/2.6.0/bin:/Users/sdobber/.gem/ruby/X.X.0/bin:/usr/local/opt/ruby/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/Library/TeX/texbin:/opt/X11/bin
  XPC_FLAGS = 0x0
  TERM = xterm-256color
```