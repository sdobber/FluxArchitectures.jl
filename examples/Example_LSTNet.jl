## Example for using LSTNet

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg;
Pkg.activate(".");
Pkg.instantiate()

@info "Loading packages"
using FluxArchitectures
using Plots

# Load some sample data
@info "Loading data"
poollength = 10
horizon = 15
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
    init=Flux.zeros32, initW=Flux.zeros32) |> gpu

# MSE loss
function loss(x, y)
    Flux.ChainRulesCore.ignore_derivatives() do
        Flux.reset!(model)
    end
    return Flux.mse(model(x), permutedims(y))
end

# Callback for plotting the training
cb = function ()
    Flux.reset!(model)
    pred = model(input) |> permutedims |> cpu
    Flux.reset!(model)
    p1 = plot(pred, label="Predict")
    p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
    display(plot(p1))
end

# Training loop
@info "Start loss" loss = loss(input, target)
@info "Starting training"
Flux.train!(loss, Flux.params(model), Iterators.repeated((input, target), 20),
    Adam(0.01), cb=cb)

@info "Finished"
@info "Final loss" loss = loss(input, target)
