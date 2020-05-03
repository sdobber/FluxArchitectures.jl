# FluxArchitectures

Complex neural network examples for Flux.jl.

When I started to dig into `Flux.jl`, almost all the examples I could find only presented small neural networks, or networks comprised of standard building bricks. This repository contains a loose collection of (slightly) more advanced network architectures, mostly centered around time series forecasting.

For some more in-depth descriptions of the different models, please also have a look at my [blog](https://sdobber.github.io/).


## Contents

* LSTNet: This "Long- and Short-term Time-series network" follows the paper by [Lai et. al.](https://arxiv.org/abs/1703.07015).

* DA-RNN: The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971).

* TPA-LSTM: The Temporal Pattern Attention LSTM network is based on the paper "Temporal Pattern Attention for Multivariate Time
Series Forecasting" by [Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf).
