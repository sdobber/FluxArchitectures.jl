## Example for using TPA-LSTM

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux, Plots, BSON
using Statistics
include("StackedLSTM.jl")
include("TPALSTM.jl")
include("../data/dataloader.jl")

# Load some sample data
poollength = 10
horizon = 6
datalength = 5000
input, target = get_data(:solar, poollength, datalength, horizon)
# Quick normalization
input = input./50 .- 0.3f0; target = target./50 .- 0.3f0

# Define the network architecture
inputsize = size(input,1)
hiddensize = 10
layers = 2
filternum = 32
filtersize = 1

# Define the neural net
model = TPALSTM(inputsize, hiddensize, poollength, layers, filternum, filtersize)

# MSE loss
function loss(x,y)
  Flux.reset!(model)
  return Flux.mse(model(x),y)
end

# Callback for plotting the training
cb = function()
  Flux.reset!(model)
  pred = model(input)'
  Flux.reset!(model)
  p1=plot(pred, label="Predict")
  p1=plot!(target, label="Data")
  display(plot(p1))
end

# Training loop
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target),10),
            ADAM(0.01), cb=cb)
