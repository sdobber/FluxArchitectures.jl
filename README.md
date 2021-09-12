# FluxArchitectures

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sdobber.github.io/FluxArchitectures.jl/stable)
[![Build Status](https://github.com/sdobber/FluxArchitectures.jl/workflows/CI/badge.svg)](https://github.com/sdobber/FluxArchitectures.jl/actions)
[![Coverage](https://codecov.io/gh/sdobber/FluxArchitectures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sdobber/FluxArchitectures.jl)


Complex neural network examples for Flux.jl.

This package contains a loose collection of (slightly) more advanced neural network architectures, mostly centered around time series forecasting.


## Installation

FluxArchitectures is *not* a registered package. To install, type `]` to activate the package manager, and use
```julia
dev https://github.com/sdobber/FluxArchitectures.jl
```
for installation. After `using FluxArchitectures`, the following functions are exported:
* `prepare_data`
* `get_data`
* `DARNN`
* `DSANet` 
* `LSTnet`
* `TPALSTM`

See their docstrings and the `examples` folder for details.


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


## Details of the Networks

### DA-RNN

The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971). See also the corresponding [blog post](https://sdobber.github.io/FA_DARNN/).

The code is based on a [PyTorch implementation](https://github.com/Seanny123/da-rnn/blob/master/modules.py) of the same model with slight adjustments.


#### Network Structure

![Model Structure Encoder](https://pic2.zhimg.com/80/v2-4e0c7c8fb419bb91a218d9a295b85fa9_1440w.jpg)
> Encoder Structure. Image from Qin et. al., "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction", [ArXiv](https://arxiv.org/abs/1704.02971), 2017.

The neural net consists of the following elements: The first part consists of an **encoder** made up of the following parts:
* A `LSTM` encoder layer. Only the hidden state is used for the output of the network.
* A feature attention layer consisting of two `Dense` layers with `tanh` activation function for the first.

The **decoder** part consist of
* A `LSTM` decoder layer. It's hidden state is used for determining a scaling of the original time series.
* A temporal attention layer operating on the hidden state of the encoder layer, consisting of two `Dense` layers similar to the encoder.
* A `Dense` layer operating on the encoder output and the temporal attention layer. Its ouput gets fed into the decoder.
* A `Dense` layer to obtain the final output based on the hidden state of the decoder.

The network layout is rather complex - please refer to the article cited above for the details.


#### Example File

The example loads some sample data and fits an DA-RNN network to the input features. Whereas the network works for larger datasets, it will just fit the mean for the sample data.


#### Known Issues

Training of the `DARNN` layer is very slow, probably due to the use of `Zygote.Buffer` in the encoder.


### DSANet

This "Dual Self-Attention Network for Multivariate Time Series Forecasting" is based on the paper by [Siteng Huang et. al.](https://kyonhuang.top/files/Huang-DSANet.pdf).

The code is based on a [PyTorch implementation](https://github.com/bighuang624/DSANet) of the same model with slight adjustments.


#### Network Structure

![Network Structure](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/DSANet-model-structure.png)

The neural net consists of the following elements:
* An autoregressive part
* A local temporal convolution mechanism, fed to a self-attention structure.
* A global temporal convolution mechanism, fed to a self-attention structure.


#### Example File

The example loads some sample data and fits an DSANet to the input features.


#### Known Issues

* The `MaxPool` pooling layers in the temporal convolution mechanisms can cause the model output to become `NaN` during training. This is not captured yet. Changing the model parameters or the random seed before training can help.

* The original model transposes some internal matrices, using `Flux.batched_transpose`. As there is no adjoint defined yet for this operation in `Zygote.jl`, we use `permutedims` instead.


### LSTNet

This "Long- and Short-term Time-series network" is based on the paper by [Lai et. al.](https://arxiv.org/abs/1703.07015). See also the corresponding [blog post](https://sdobber.github.io/FA_LSTNet/).


#### Network Structure

![Model Structure](https://opringle.github.io/images/model_architecture.png)
> Image from Lai et al, "Long- and Short-term Time-series network", [ArXiv](https://arxiv.org/abs/1703.07015) 2017.

The neural net consists of the following elements:
* A convolutional layer than operates on some window of the time series.
* A `GRU` cell with `relu` activation function.
* A `SkipGRU` similar to the previous `GRU` cell, with the difference that the hidden state is taken from a specific amount of timesteps back in time. Both the `GRU` and the `SkipGRU` layer take their input from the convolutional layer.
* A dense layer that operates on the concatenated output of the previous two layers.
* An autoregressive layer operating on the input data itself, being added to the output of the dense layer.


#### Example File

The example loads some sample data and fits an LSTnet to the input features.


### TPA-LSTM

![Model Structure](https://miro.medium.com/max/1400/1*SjKMs_iTOJaKqx45fpYEDQ.png)
> Image from Shih et. al., "Temporal Pattern Attention for Multivariate Time Series Forecasting", [ArXiv](https://arxiv.org/pdf/1809.04206v2.pdf), 2019.

The Temporal Pattern Attention LSTM network is based on the paper "Temporal Pattern Attention for Multivariate Time
Series Forecasting" by [Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf). See also the corresponding [blog post](https://sdobber.github.io/FA_TPALSTM/).

The code is based on a [PyTorch implementation](https://github.com/jingw2/demand_forecast/blob/master/tpa_lstm.py) by Jing Wang of the same model with slight adjustments.


#### Network Structure

The neural net consists of the following elements: The first part consists of an **embedding** and **stacked LSTM layer** made up of the following parts:
* A `Dense` embedding layer for the input data.
* A `StackedLSTM` layer for the transformed input data.

The **temporal attention mechanism** consist of
* A `Dense` layer that transforms the hidden state of the last LSTM layer in the `StackedLSTM`.
* A convolutional layer operating on the pooled output of the previous layer, estimating the importance of the different datapoints.
* A `Dense` layer operating on the LSTM hidden state and the output of the attention mechanism.

A final `Dense` layer is used to calculate the output of the network.


#### Stacked LSTM

The stacked version of a number of `LSTM` cells is obtained by feeding the hidden state of one cell as input to the next one. `Flux.jl`'s standard setup only allows feeding the output of one cell as the new input, thus we adjust some of the internals:
* Management of hidden states in `Flux` is done by the `Recur` structure, which returns the output of a recurrent layer. We use a `HiddenRecur` structure instead which returns the hidden state.
* The `StackedLSTM`-function chains everything together depending on the number of layers. (One layer corresponds to a standard `LSTM` cell.)


#### Example File

The example loads some sample data and fits a TPA-LSTM network to the input features. Whereas the network works for larger datasets, it will just fit the mean for the sample data.



