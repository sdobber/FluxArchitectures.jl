# Benchmarks.jl
#
# Run some benchmarks for FluxArchitectures

# Settings
const RUN_GPU = false
const RUN_CPU = true

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

@info "Loading packages"
using Flux, BSON, BenchmarkTools, DataFrames
using SliceMap, JuliennedArrays, Random
include("../shared/Sequentialize.jl")
include("../data/dataloader.jl")
include("../DARNN/DARNN.jl")
include("../DSANet/DSANet.jl")
include("../LSTNet/LSTNet.jl")
include("../TPA-LSTM/TPALSTM.jl")

# Initialize results dataframe
results = DataFrame(Name=String[], CPU=Union{Missing,Float32}[], GPU=Union{Missing,Float32}[])

# Set up data and loss
poollength = 10
horizon = 6
datalength = 500
input_cpu, target_cpu = get_data(:solar, poollength, datalength, horizon)
input_gpu, target_gpu = gpu(input_cpu), gpu(target_cpu)
inputsize = size(input_cpu, 1)
function loss(x, y)
    Flux.reset!(model)
    return Flux.mse(model(x), y')
end


## DARNN ----------------------------------
@info "DARNN"
push!(results,["DARNN", missing, missing])
encodersize = 10
decodersize = 10
if RUN_GPU
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1) |> gpu
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_gpu, target_gpu), 5),
            ADAM(0.007))
    results[end,:GPU] = time
end
if RUN_CPU
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_cpu, target_cpu), 5),
            ADAM(0.007))
    results[end,:CPU] = time
end


## DSANet ----------------------------------
@info "DSANet"
push!(results,["DSANet", missing, missing])
local_length = 3
n_kernels = 3
d_model = 4
hiddensize = 1
n_layers = 3
n_head = 2
if RUN_GPU
    Random.seed!(123)
    model = DSANet(inputsize, poollength, local_length, n_kernels, d_model,
               hiddensize, n_layers, n_head) |> gpu
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_gpu, target_gpu), 5),
            ADAM(0.007))
    results[end,:GPU] = time
end
if RUN_CPU
    Random.seed!(123)
    model = DSANet(inputsize, poollength, local_length, n_kernels, d_model,
               hiddensize, n_layers, n_head)
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_cpu, target_cpu), 5),
            ADAM(0.007))
    results[end,:CPU] = time
end


## LSTNet ----------------------------------
@info "LSTNet"
push!(results,["LSTNet", missing, missing])
convlayersize = 2
recurlayersize = 3
skiplength = 120
if RUN_GPU
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
        init=Flux.zeros, initW=Flux.zeros) |> gpu
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_gpu, target_gpu), 5),
            ADAM(0.007))
    results[end,:GPU] = time
end
if RUN_CPU
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
        init=Flux.zeros, initW=Flux.zeros)
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_cpu, target_cpu), 5),
            ADAM(0.007))
    results[end,:CPU] = time
end


## TPA-LSTM ----------------------------------
@info "TPA-LSTM"
push!(results,["TPA-LSTM", missing, missing])
hiddensize = 10
layers = 2
filternum = 32
filtersize = 1
if RUN_GPU
    model = TPALSTM(inputsize, hiddensize, poollength, layers, filternum, filtersize) |> gpu
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_gpu, target_gpu), 5),
            ADAM(0.007))
    results[end,:GPU] = time
end
if RUN_CPU
    model = TPALSTM(inputsize, hiddensize, poollength, layers, filternum, filtersize)
    time = @belapsed Flux.train!(loss, Flux.params(model),Iterators.repeated((input_cpu, target_cpu), 5),
            ADAM(0.007))
    results[end,:CPU] = time
end