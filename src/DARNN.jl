# DARNN.jl
#
# Layers for a DA-RNN network
# Based on paper Quin et. al. "A Dual-Stage Attention-Based Recurrent Neural
# Network for Time Series Prediction" https://arxiv.org/abs/1704.02971

# Struct that stores layers and some extra information
mutable struct DARNNCell{A,B,C,D,E,F,W,X,Y,Z}
    # Encoder part
    encoder_lstm::A
    encoder_attn::B
    # Decoder part
    decoder_lstm::C
    decoder_attn::D
    decoder_fc::E
    decoder_fc_final::F
    # Index for original data etc
    encodersize::W
    decodersize::X
    orig_idx::Y
    poollength::Z
end

# Initialization function to obtain the right size of the hidden state
# after a `reset!` of the model.
function darnn_init(m::DARNNCell, x)
    s = size(x)
    m.encoder_lstm(simarray(x, s[1], s[4]))
    m.decoder_lstm(simarray(x, 1, s[4]))
    return nothing
end
simarray(x::CUDA.CuArray, size...) = CUDA.zeros(eltype(x), size...)
simarray(x, size...) = zeros(eltype(x), size...)

Flux.ChainRulesCore.@non_differentiable darnn_init(m, x)

# Dispatch on `trainable` and `reset!` to make standard Flux scripts work
Flux.trainable(m::DARNNCell) = (m.encoder_lstm, m.encoder_attn, m.decoder_lstm,
    m.decoder_attn, m.decoder_fc, m.decoder_fc_final)
Flux.reset!(m::DARNNCell) = foreach(Flux.reset!, (m.encoder_lstm, m.decoder_lstm))
Flux.@functor DARNNCell

"""
    DARNN(inp, encodersize, decodersize, poollength, orig_idx)

Create a DA-RNN layer based on the architecture described in
[Qin et. al.](https://arxiv.org/abs/1704.02971), as implemented for PyTorch
[here](https://github.com/Seanny123/da-rnn/blob/master/modules.py). `inp` specifies the
number of input features. `encodersize` defines the number of LSTM encoder layers, and
`decodersize` defines the number of LSTM decoder layers. `poolsize` gives the
length of the window for the pooled input data, and `orig_idx` defines the array
index where the original time series is stored in the input data,

Data is expected as array with dimensions `features x poolsize x 1 x data`, i.e.
for 1000 data points containing 31 features that have been windowed over 6
timesteps, `DARNN` expects an input size of `(31, 6, 1, 1000)`.

Takes the keyword arguments `init` and `bias` for the initialization of the
weight vector and bias of the linear layers.
"""
function DARNN(inp::Integer, encodersize::Integer, decodersize::Integer, poollength::Integer, orig_idx::Integer;
    init=Flux.glorot_uniform, bias=true)

    # Encoder part
    encoder_lstm = Seq(HiddenRecur(Flux.LSTMCell(inp, encodersize)))
    encoder_attn = Chain(Dense(2 * encodersize + poollength, poollength, init=init, bias=bias),
        a -> tanh.(a),
        Dense(poollength, 1, init=init, bias=bias))
    # Decoder part
    decoder_lstm = Seq(HiddenRecur(Flux.LSTMCell(1, decodersize)))
    decoder_attn = Chain(Dense(2 * decodersize + encodersize, encodersize, init=init, bias=bias),
        a -> tanh.(a),
        Dense(encodersize, 1, init=init, bias=bias))
    decoder_fc = Dense(encodersize + 1, 1)
    decoder_fc_final = Dense(decodersize + encodersize, 1, init=init, bias=bias)

    return DARNNCell(encoder_lstm, encoder_attn, decoder_lstm, decoder_attn, decoder_fc,
        decoder_fc_final, encodersize, decodersize, orig_idx, poollength)
end

# model output
function (m::DARNNCell)(x)
    # Initialize after reset
    size(m.encoder_lstm.state, 1) == 1 && darnn_init(m, x)
    input_data = dropdims(x; dims=3)
    input_encoded = darnn_encoder(m, input_data)
    context = darnn_decoder(m, input_encoded, input_data)
    return m.decoder_fc_final(cat(m.decoder_lstm.state[1], context, dims=1))
end

function _encoder(m::DARNNCell, input_data, slice)
    hidden = m.encoder_lstm.state[1]
    cell = m.encoder_lstm.state[2]
    repeatmat = darnn_getones(input_data)

    x = cat(hidden .* repeatmat, cell .* repeatmat,  # CUDA compatible repeat(hidden, inner=(1,1,size(input_data,1)) etc.
        permutedims(input_data, [2, 3, 1]), dims=1) |>
        a -> m.encoder_attn(reshape(a, (:, size(input_data, 1) * size(input_data, 3))))
    attn_weights = Flux.softmax(reshape(x, (size(input_data, 1), size(input_data, 3))))
    weighted_input = attn_weights .* slice
    _ = m.encoder_lstm(weighted_input)
    return m.encoder_lstm.state[1]
end

function darnn_encoder(m::DARNNCell, input_data::Array)
    sl = JuliennedArrays.Slices(input_data, True(), False(), True())
    fun = s -> _encoder(m, input_data, s)
    return Align(map(fun, sl), True(), False(), True())
end

function darnn_encoder(m::DARNNCell, input_data::CUDA.CuArray)
    input_encoded = Flux.Zygote.Buffer(input_data, m.encodersize, m.poollength,
        size(input_data, 3))
    @inbounds for t in 1:m.poollength
        input_encoded[:, t, :] = Flux.unsqueeze(_encoder(m, input_data, input_data[:, t, :]), dims=2)
    end
    return copy(input_encoded)
end

function darnn_decoder(m::DARNNCell, input_encoded, input_data)
    context = Flux.zeros32(m.encodersize, size(input_data, 3))
    @inbounds for t in 1:m.poollength
        hidden = m.decoder_lstm.state[1]
        cell = m.decoder_lstm.state[2]
        repeatmat = darnn_getones(input_data, m.poollength)
        x = cat(permutedims(hidden .* repeatmat, [1, 3, 2]),
            permutedims(cell .* repeatmat, [1, 3, 2]),
            input_encoded, dims=1) |>
            a -> m.decoder_attn(reshape(a, (2 * m.decodersize + m.encodersize, :))) |>
                 a -> Flux.softmax(reshape(a, (m.poollength, :)))
        context = dropdims(NNlib.batched_mul(input_encoded, Flux.unsqueeze(x, dims=2)), dims=2)
        ỹ = m.decoder_fc(cat(context, input_data[m.orig_idx, t, :]', dims=1))
        _ = m.decoder_lstm(ỹ)
    end
    return context
end

# pretty printing
function Base.show(io::IO, l::DARNNCell)
    print(io, "DARNN(", size(l.encoder_lstm.chain.cell.Wi, 2))
    print(io, ", ", l.encodersize)
    print(io, ", ", Int(size(l.decoder_lstm.chain.cell.Wi, 1) / 4))
    print(io, ", ", l.poollength)
    print(io, ", ", l.orig_idx)
    print(io, ")")
end

# Helper functions to replace `repeat` with CUDA friendly version
darnn_getones(x::Array, size) = Flux.ones32(1, 1, size)
darnn_getones(x::CUDA.CuArray, size) = CUDA.ones(1, 1, size)
darnn_getones(x) = darnn_getones(x, size(x, 1))
Flux.ChainRulesCore.@non_differentiable darnn_getones(x, sz)
Flux.ChainRulesCore.@non_differentiable darnn_getones(x)
