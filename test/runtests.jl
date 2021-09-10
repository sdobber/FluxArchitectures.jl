using FluxArchitectures
using Test

if Flux.use_cuda[]
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