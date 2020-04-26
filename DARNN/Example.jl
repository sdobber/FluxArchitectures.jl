## Example for using LSTNet

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux, Plots, BSON
using Statistics
include("DARNN.jl")

# Load some sample data
BSON.@load "../data/SampleData.bson" input target

# Define the network architecture
inputsize = size(input,1)
encodersize = 10
decodersize = 10
poolrows=6

# Define the neural net
model = DARNN(inputsize, encodersize, decodersize, poolrows, 25)

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
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target),50),
            ADAM(0.007), cb=cb)
