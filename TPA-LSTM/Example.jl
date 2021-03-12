## Example for using TPA-LSTM

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

@info "Loading packages"
using Flux, BSON, Plots
using SliceMap, JuliennedArrays
include("../shared/Sequentialize.jl")
include("../data/dataloader.jl")
include("TPALSTM.jl")

# Load some sample data
@info "Loading data"
poollength = 10
horizon = 6
datalength = 2000
input, target = get_data(:solar, poollength, datalength, horizon) |> gpu

# Define the network architecture
@info "Creating model and loss"
inputsize = size(input, 1)
hiddensize = 10
layers = 2
filternum = 32
filtersize = 1

# Define the neural net
model = TPALSTM(inputsize, hiddensize, poollength, layers, filternum, filtersize) |> gpu

# MSE loss
function loss(x, y)
    Flux.reset!(model)
    return Flux.mse(model(x), y')
end

# Callback for plotting the training
cb = function ()
    Flux.reset!(model)
    pred = model(input)' |> cpu
    Flux.reset!(model)
    p1 = plot(pred, label="Predict")
    p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
    display(plot(p1))
end

# Training loop
@info "Start loss" loss = loss(input, target)
@info "Starting training"
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 50),
            ADAM(0.02), cb=cb)

@info "Finished"
@info "Final loss" loss = loss(input, target)