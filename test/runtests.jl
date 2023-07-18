using FluxArchitectures
using CUDA, cuDNN
using Test

# Solution from https://github.com/JuliaLang/julia/issues/18780
struct NoException <: Exception end
macro test_nothrow(ex)
    esc(:(@test_throws NoException ($(ex); throw(NoException()))))
end

# Do a forward pass
function fw_gpu(m, ip)
    CUDA.@sync m(ip)
end
function fw_cpu(m, ip)
    m(ip)
end

# Do a forward + backward pass
function bw_gpu(m, ip)
    gs = CUDA.@sync gradient((m, x) -> sum(m(x)), m, ip)
end
function bw_cpu(m, ip)
    gs = gradient((m, x) -> sum(m(x)), m, ip)
end

if CUDA.functional()
    @info "Testing GPU support"
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "DARNN" begin
    include("DARNN.jl")
end

@testset "DSANet" begin
    include("DSANet.jl")
end

@testset "LSTnet" begin
    include("LSTnet.jl")
end

@testset "TPALSTM" begin
    include("TPALSTM.jl")
end

@testset "dataloader" begin
    include("dataloader.jl")
end

@testset "shared" begin
    include("shared.jl")
end