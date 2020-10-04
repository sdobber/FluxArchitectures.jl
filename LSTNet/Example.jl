## Example for using LSTNet

# Make sure all the required packages are available
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux, Plots, BSON
include("../shared/Sequentialize.jl")
include("../data/dataloader.jl")
include("LSTnet.jl")

# Load some sample data
poollength = 10
horizon = 6
datalength = 1000
input, target = get_data(:exchange_rate, poollength, datalength, horizon)

# Define the network architecture
inputsize = size(input,1)
convlayersize = 2
recurlayersize = 3
skiplength = 120

# Define the neural net
model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)

# MSE loss
function loss(x,y)
  Flux.reset!(model)
  return sum(abs2,model(x) - y')
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
            ADAM(0.01), cb=cb)
