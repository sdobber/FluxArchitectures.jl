## Example for using DSANet

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate(".")
Pkg.instantiate()

using Flux, Plots, BSON
using Random
include("DSANet.jl")
include("../data/dataloader.jl")

# Load some sample data
poollength = 10
horizon = 6
datalength = 4000
input, target = get_data(:exchange_rate, poollength, datalength, horizon) |> gpu

# Define the network architecture
inputsize = size(input,1)
local_length = 3
n_kernels = 3
d_model = 4
hiddensize = 1
n_layers = 3
n_head = 2

# Define the neural net
Random.seed!(123)
model = DSANet(inputsize, poollength, local_length, n_kernels, d_model,
               hiddensize, n_layers, n_head) |> gpu

# MSE loss
function loss(x,y)
  Flux.reset!(model)
  return Flux.mse(model(x),y')
end

# Callback for plotting the training
cb = function()
  Flux.reset!(model)
  pred = model(input)' |> cpu
  Flux.reset!(model)
  p1=plot(pred, label="Predict")
  p1=plot!(cpu(target), label="Data", title="Loss $(loss(input,target))")
  display(plot(p1))
end

# Training loop
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target),50),
            ADAM(0.005), cb=cb)
