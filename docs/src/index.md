```@meta
CurrentModule = FluxArchitectures
```

# FluxArchitectures

Documentation for [FluxArchitectures](https://github.com/sdobber/FluxArchitectures.jl).


## Installation

Download [Julia 1.6](http://www.julialang.org) or later, if you haven't already. You can add FluxArchitectures from  Julia's package manager, by typing 
```
] add FluxArchitectures
``` 
in the Julia prompt.


## Models

* [LSTnet](@ref): This "Long- and Short-term Time-series network" follows the paper by [Lai et. al.](https://arxiv.org/abs/1703.07015).

* [DARNN](@ref): The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971).

* [TPA-LSTM](@ref): The Temporal Pattern Attention LSTM network is based on the paper "Temporal Pattern Attention for Multivariate Time Series Forecasting" by [Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf).

* [DSANet](@ref): The "Dual Self-Attention Network for Multivariate Time Series Forecasting" is based on the paper by [Siteng Huang et. al.](https://kyonhuang.top/files/Huang-DSANet.pdf)


## Quickstart

Activate the package and load some sample-data:
```julia
using FluxArchitectures
poollength = 10; horizon = 6; datalength = 1000;
input, target = get_data(:exchange_rate, poollength, datalength, horizon) 
```

Define a model and a loss function:
```julia
model = LSTnet(size(input, 1), 2, 3, poollength, 120)
loss(x, y) = Flux.mse(model(x), y')
```

Train the model:
```julia
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 20), ADAM(0.01))
```
