# FluxArchitectures

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sdobber.github.io/FluxArchitectures.jl/dev)
[![Build Status](https://github.com/sdobber/FluxArchitectures.jl/workflows/CI/badge.svg)](https://github.com/sdobber/FluxArchitectures.jl/actions)
[![Coverage](https://codecov.io/gh/sdobber/FluxArchitectures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sdobber/FluxArchitectures.jl)


Complex neural network examples for Flux.jl.

This package contains a loose collection of (slightly) more advanced neural network architectures, mostly centered around time series forecasting.


## Installation

To install FluxArchitectures, type `]` to activate the package manager, and type
```julia
add FluxArchitectures
```
for installation. After `using FluxArchitectures`, the following functions are exported:
* `prepare_data`
* `get_data`
* `DARNN`
* `DSANet` 
* `LSTnet`
* `TPALSTM`

See their docstrings, the [documentation]((https://sdobber.github.io/FluxArchitectures.jl/stable)), and the `examples` folder for details.


## Models

* **LSTnet**: This "Long- and Short-term Time-series network" follows the paper by [Lai et. al.](https://arxiv.org/abs/1703.07015).

* **DARNN**: The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971).

* **TPA-LSTM**: The Temporal Pattern Attention LSTM network is based on the paper "Temporal Pattern Attention for Multivariate Time Series Forecasting" by [Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf).

* **DSANet**: The "Dual Self-Attention Network for Multivariate Time Series Forecasting" is based on the paper by [Siteng Huang et. al.](https://kyonhuang.top/files/Huang-DSANet.pdf)


## Quickstart

Activate the package and load some sample-data:
```julia
using FluxArchitectures
poollength = 10; horizon = 15; datalength = 1000;
input, target = get_data(:exchange_rate, poollength, datalength, horizon) 
```

Define a model and a loss function:
```julia
model = LSTnet(size(input, 1), 2, 3, poollength, 120)
loss(x, y) = Flux.mse(model(x), y')
```

Train the model:
```julia
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 20), Adam(0.01))
```
