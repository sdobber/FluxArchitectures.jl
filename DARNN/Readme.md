# DA-RNN

The "Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" is based on the paper by [Qin et. al.](https://arxiv.org/abs/1704.02971). See also the corresponding [blog post](https://sdobber.github.io/FA_DARNN/).

The code is based on a [PyTorch implementation](https://github.com/Seanny123/da-rnn/blob/master/modules.py) of the same model with slight adjustments.


## Network Structure

The neural net consists of the following elements: The first part consists of an **encoder** made up of the following parts:
* A `LSTM` encoder layer. Only the hidden state is used for the output of the network.
* A feature attention layer consisting of two `Dense` layers with `tanh` activation function for the first.

The **decoder** part consist of
* A `LSTM` decoder layer. It's hidden state is used for determining a scaling of the original time series.
* A temporal attention layer operating on the hidden state of the encoder layer, consisting of two `Dense` layers similar to the encoder.
* A `Dense` layer operating on the encoder output and the temporal attention layer. Its ouput gets fed into the decoder.
* A `Dense` layer to obtain the final output based on the hidden state of the decoder.

The network layout is rather complex - please refer to the article cited above for the details.


## Example File

The example loads some sample data and fits an DA-RNN network to the input features. Whereas the network works for larger datasets, it will just fit the mean for the sample data.


## Known Issues

Training of the `DARNN` layer is very slow, probably due to the use of `Zygote.Buffer` in the encoder.
