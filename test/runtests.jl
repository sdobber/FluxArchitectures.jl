using FluxArchitectures
using Test

macro no_error(ex)
    quote
        try
            $(esc(ex))
            true
        catch
            false
        end
    end
end

# Do a backward pass
function bw_gpu(m, inp)
    gs = Flux.CUDA.@sync gradient((m, x) -> sum(m(x)), m, inp)
end
function bw_cpu(m, inp)
    gs = gradient((m, x) -> sum(m(x)), m, inp)
end

if Flux.CUDA.functional()
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