## Example for using LSTNet

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

@info "Loading packages"
using Flux, BSON, Plots
using SliceMap, JuliennedArrays
include("../shared/Sequentialize.jl")
include("../data/dataloader.jl")
include("LSTnet.jl")

# Load some sample data
@info "Loading data"
poollength = 10
horizon = 6
datalength = 1000
input, target = get_data(:exchange_rate, poollength, datalength, horizon) |> gpu

# Define the network architecture
@info "Creating model and loss"
inputsize = size(input, 1)
convlayersize = 2
recurlayersize = 3
skiplength = 120

# Define the neural net
model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
        init=Flux.zeros, initW=Flux.zeros, initb=Flux.zeros) |> gpu

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
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 20),
            ADAM(0.01), cb=cb)

@info "Finished"
@info "Final loss" loss = loss(input, target)
