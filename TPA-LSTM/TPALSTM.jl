# TPALSTM.jl
#
# Layers for TPA-LSTM

mutable struct TPALSTMCell{A, B, C, D, E, F, G, X, Y, Z}
    # Prediction layers
    embedding::A
    output::B
    lstm::C
    # Attention layers
    attention_linear1::D
    attention_linear2::E
    attention_conv::F
    attention_features_size::G
    # Metadata
    filternum::X
    poollength::Y
    hiddensize::Z
end

"""
	TPALSTM(in, hiddensize, poollength)
	TPALSTM(in, hiddensize, poollength, layers, filternum, filtersize)

Creates a TPA-LSTM layer based on the architecture described in
[Shih et. al.](https://arxiv.org/pdf/1809.04206v2.pdf), as implemented for PyTorch by
[Jing Wang](https://github.com/jingw2/demand_forecast/blob/master/tpa_lstm.py). `in`
specifies the number of input features. `hiddensize` defines the input and output size of
the LSTM layer, and `layers` the number of LSTM layers (with standard value `1`).
`filternum` and `filtersize` define the number and size of filters in the
attention layer. Standard values are `32` and `1`. `poolsize` gives the length of the window
for the pooled input data.

Data is expected as array with dimensions `features x poolsize x 1 x data`, i.e.
for 1000 data points containing 31 features that have been windowed over 6
timesteps, `TPALSTM` expects an input size of `(31, 6, 1, 1000)`.

Takes the keyword arguments `initW` and `initb` for the initialization of the
`Dense` layers, and `init` for the initialization of the `StackedLSTM` network.
"""
function TPALSTM(in::Integer, hiddensize::Integer, poollength::Integer, layers=1,
	 			filternum=32, filtersize=1; initW = Flux.glorot_uniform, initb = Flux.zeros,
				init = Flux.glorot_uniform)

	embedding = Dense(in, hiddensize, Flux.relu; initW = initW, initb = initb)
    output = Dense(hiddensize, 1; initW = initW, initb = initb)  # 1 could be replaced by output_horizon
    lstm = Seq(StackedLSTM(hiddensize, hiddensize, hiddensize, layers; init = init))

    attention_length = poollength - 1
    attention_size = hiddensize
    attention_feature_size = attention_size - filtersize + 1
    channels = poollength - 1
    attention_linear1 = Dense(attention_size, filternum; initW = initW, initb = initb)
    attention_linear2 = Dense(attention_size + filternum, attention_size;
							  initW = initW, initb = initb)
    attention_conv = Conv((filtersize, attention_length), 1 => filternum)

    return TPALSTMCell(embedding, output, lstm, attention_linear1, attention_linear2,
                    attention_conv, attention_feature_size, filternum, poollength,
                    hiddensize)
end

# Define parameters and resetting the LSTM layer
Flux.trainable(m::TPALSTMCell) = (m.embedding, m.output, m.lstm, m.attention_linear1,
 				m.attention_linear2, m.attention_conv)
Flux.reset!(m::TPALSTMCell) = Flux.reset!(m.lstm.chain)
Flux.@functor TPALSTMCell # embedding, output, lstm, attention_linear1, attention_linear2, attention_conv, 

# Attention part of the network
function _TPALSTM_attention(H, h_last, m::TPALSTMCell)
    H_u = Flux.unsqueeze(H, 3)
    conv_vecs = Flux.relu.(dropdims(m.attention_conv(H_u), dims=2))
    w = m.attention_linear1(h_last) |>
            a -> Flux.unsqueeze(a, 1)
    alpha = Flux.sigmoid.(sum(conv_vecs.*w, dims=2))
    v = dropdims(sum(alpha.*conv_vecs, dims=1), dims=1)
    concat = cat(h_last, v, dims=1)
    return m.attention_linear2(concat)
end

# Get the pooled hidden state from feeding it to the LSTM cell
# TO DO: Get rid of Zygote.Buffer for speeding things up
# function _TPALSTM_gethidden(inp, m::TPALSTMCell)
#     batchsize = size(inp,3)
#     H = Flux.Zygote.Buffer([0.0f0], m.hiddensize, m.poollength-1, batchsize)  # needs to be off GPU
#     @inbounds for t in 1:m.poollength-1
#         x = inp[:,t,:]
#         xconcat = m.embedding(x)
#         hiddenstate = m.lstm(xconcat)
#         H[:,t,:] .= hiddenstate 
#     end
#     return copy(H)  # returns CPU array - transfer to GPU again
# end
function _TPALSTM_gethidden(inp, m::TPALSTMCell)
        x = Flux.unstack(inp, 2)[1:end-1]
        xconcat = m.embedding.(x)
        hiddenstate = m.lstm.(xconcat)
        H = Flux.stack(hiddenstate,2)
    return H 
end

# Model output
function (m::TPALSTMCell)(x)
    inp = dropdims(x, dims=3)
    H_raw = _TPALSTM_gethidden(inp, m)
	H = Flux.relu.(H_raw)
    x = inp[:,end,:]
    xconcat = m.embedding(x)
    h_last = m.lstm(xconcat)
    ht = _TPALSTM_attention(H, h_last, m)
    return m.output(ht)
end

# Pretty printing
function Base.show(io::IO, l::TPALSTMCell)
	print(io, "TPALSTM(", size(l.embedding.W,2))
	print(io, ", ", l.hiddensize)
	print(io, ", ", l.poollength)
	length(l.lstm.chain.chain) == 1 || print(io, ", ", length(l.lstm.chain.chain))
	(l.filternum == 32 && size(l.attention_conv.weight,1) == 1) || print(io, ", ", l.filternum)
	(l.filternum == 32 && size(l.attention_conv.weight,1) == 1) || print(io, ", ", size(l.attention_conv.weight,1))
	print(io, ")")
end

# Initialize forget gate bias with 1
initialize_bias!(l::TPALSTMCell) = initialize_bias!(l.lstm.chain)
