# FluxArchitectures

Complex neural network examples for Flux.jl.

When I started to dig into `Flux.jl`, almost all the examples I could find only presented small neural networks, or networks comprised of standard building bricks. This repository contains a loose collection of (slightly) more advanced network architectures, mostly centered around time series forecasting.

For some more in-depth descriptions of the different models, please also have a look at my [blog](https://sdobber.github.io/).


## Contents

* LSTNet: This "Long- and Short-term Time-series network" follows the paper by [Lai et. al.](https://arxiv.org/abs/1703.07015).

* DA-RNN: The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971).

* TPA-LSTM: The Temporal Pattern Attention LSTM network is based on the paper "Temporal Pattern Attention for Multivariate Time
Series Forecasting" by [Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf).

* DSANet: The "Dual Self-Attention Network for Multivariate Time Series Forecasting" is based on the paper by [Siteng Huang et. al.](https://kyonhuang.top/files/Huang-DSANet.pdf)


## Example data

Currently, the following example data from https://github.com/laiguokun/multivariate-time-series-data is included:

* `:solar`: The raw data is coming from http://www.nrel.gov/grid/solar-power-data.html: It contains the solar power production records in the year of 2006, which is sampled every 10 minutes from 137 PV plants in Alabama State.

* `:traffic`: The raw data is coming from http://pems.dot.ca.gov. The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.

* `:electricity`: The raw dataset is from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014 for 321 clients. The data has been cleaned and converted to hourly consumption.

* `:exchange_rate`: The collection of daily exchange rates of eight foreign countries including Australia, Great Britain, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.


## A Note on GPU Calculations

Finally, GPU support for all models has been added! However, it cannot be guaranteed that this is done in an optimal way. Currently, only `DSANet` sees an improvement in training speed on my hardware, whereas all other models train faster on a CPU. The deeper reason is the current limitation of Flux's implementation of RNNs, see this issue https://github.com/FluxML/Flux.jl/issues/1365 that affects all models except `DASNet`, which doesn't use recurrence. 