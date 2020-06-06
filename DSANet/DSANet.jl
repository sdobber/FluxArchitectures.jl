# DSANet.jl
#
# Layers for a DSANet network
# Based on paper Siteng Huang et. al. "DSANet: Dual Self-Attention Network for Multivariate
# Time Series Forecasting" https://kyonhuang.top/files/Huang-DSANet.pdf

# Helper Functions
"""
    Scaled_Dot_Product_Attention(q, k, v, temperature)

Scaled dot product attention function with query `q`, keys `k` and values `v`. Normalisation
is given by `temperature`. Outputs
``\\mathrm{softmax}\\left( \\frac{q \\cdot k^T}{\\mathrm{temperature}} \\right)\\cdot v``.
"""
function Scaled_Dot_Product_Attention(q, k, v, temperature, attn_dropout=0.1)
    #attn1 = NNlib.batched_mul(q, Flux.batched_transpose(k)) / temperature
	attn1 = NNlib.batched_mul(q, permutedims(k,[2,1,3])) / temperature
	attn2 = Flux.softmax(attn1, dims=2)
    attn3 = Dropout(attn_dropout)(attn2)
    return NNlib.batched_mul(attn3,v)
end

function regularized_normalise(x::AbstractArray; dims=1)
  μ′ = Flux.mean(x, dims = dims)
  σ′ = Flux.std(x, dims = dims, mean = μ′, corrected=false)
  ϵ = sqrt(eps(Float32))
  return (x .- μ′) ./ (σ′ .+ ϵ)
end

"""
    Reg_LayerNorm(h::Integer)
A [normalisation layer](https://arxiv.org/pdf/1607.06450.pdf) designed to be
used with recurrent hidden states of size `h`. Normalises the mean and standard
deviation of each input before applying a per-neuron gain/bias. To avoid numeric
overflow, the division by the standard deviation has been regularised by
adding `ϵ = 1E-5`.
"""
struct Reg_LayerNorm{T}
  diag::Flux.Diagonal{T}
end

Reg_LayerNorm(h::Integer) =
  Reg_LayerNorm(Flux.Diagonal(h))

Flux.@functor Reg_LayerNorm

(a::Reg_LayerNorm)(x) = a.diag(regularized_normalise(x))

function Base.show(io::IO, l::Reg_LayerNorm)
  print(io, "Reg_LayerNorm(", length(l.diag.α), ")")
end


# Encoder structure
mutable struct SelfAttn_EncoderCell{A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P}
    # General
    dropout::A
    # Multi head attention
    layer_norm::B  # currently not used
    mha_w_qs::C
    mha_w_ks::D
    mha_w_vs::E
    mha_fc::F
    # Positionwise Feedforward
    pwff_w_1::G
    pwff_w_2::H
	pwff_layer_norm::P
    # Metadata
    inp::I
    n_head::J
    d_model::K
    d_hid::L
    d_k::M
    d_v::N
    σ::O
end

Flux.@functor SelfAttn_EncoderCell
Flux.trainable(m::SelfAttn_EncoderCell) = (m.layer_norm, m.mha_w_qs, m.mha_w_ks, m.mha_w_vs,
            m.mha_fc, m.pwff_w_1, m.pwff_w_2, m.pwff_layer_norm)

"""
    SelfAttn_Encoder(inp, d_model, n_head, d_hid, drop_prob = 0.1f0, σ = Flux.relu)

Encoder part for the self attention networks that comprise the `DSANet`. For parameters see
[DSANet](@ref).
"""
function SelfAttn_Encoder(inp, d_model, n_head, d_hid, drop_prob = 0.1f0, σ = Flux.relu)
    (d_model % n_head == 0) || throw(ArgumentError("d_model = $d_model must be divisible by n_head = $n_head"))

    d_k = d_model ÷ n_head
    d_v = d_model ÷ n_head

    # Multi head attention
    mha_w_qs = Dense(d_model, n_head * d_k)
    mha_w_ks = Dense(d_model, n_head * d_k)
    mha_w_vs = Dense(d_model, n_head * d_v)
    mha_fc = Dense(n_head*d_v, d_model)
    layer_norm = Reg_LayerNorm(d_model)
    # Positionwise Feedforward
    pwff_w_1 = Conv((1,), d_model => d_hid)
    pwff_w_2 = Conv((1,), d_hid => d_model)
	pwff_layer_norm = Reg_LayerNorm(d_model)

    return SelfAttn_EncoderCell(Dropout(drop_prob), layer_norm, mha_w_qs, mha_w_ks, mha_w_vs, mha_fc,
                        pwff_w_1, pwff_w_2, pwff_layer_norm, inp, n_head, d_model, d_hid, d_k, d_v, σ)
end

function Base.show(io::IO, m::SelfAttn_EncoderCell)
	print(io, "SelfAttn_Encoder(", m.inp)
	print(io, ", ", m.d_model)
	print(io, ", ", m.n_head)
	print(io, ", ", m.d_hid)
	m.dropout.p == 0.1 || print(io, ", ", m.dropout.p)
	m.σ == Flux.relu || print(io, ", ", m.σ)
	print(io, ")")
end

function (m::SelfAttn_EncoderCell)(src_seq)
    # Multi head attention
    q = k = v = src_seq
    mha_residual = q
    q1 = reshape(q, (m.d_model, :))
    q2 = reshape(m.mha_w_qs(q1), (m.n_head, m.d_k, :) )
	k1 = reshape(k, (m.d_model, :))
	k2 = reshape(m.mha_w_ks(k1), (m.n_head, m.d_k,:) )
	v1 = reshape(v, (m.d_model, :))
	v2 = reshape(m.mha_w_ks(v1), (m.n_head, m.d_v,:) )

    q3 = permutedims(q2,[1,3,2])
    k3 = permutedims(k2,[1,3,2])
    v3 = permutedims(v2,[1,3,2])

    output1 = Scaled_Dot_Product_Attention(q3, k3, v3, sqrt(Float32(m.d_k)))
	output2 = permutedims(output1, [1,3,2])
    output3 = reshape(output2, (m.d_model, :))
    output4 = m.mha_fc(output3)
    output5 = reshape(output4,(m.d_model, m.inp,:))
    output6 = m.dropout(output5)
	# this line crashes the training by creating NaNs in the Conv layer parameters
	# output7 = m.layer_norm(output6 .+ mha_residual)

    # Positionwise feedforward
    pwff_residual = output6 .+ mha_residual  # output7
    #output1 = Flux.batched_transpose(pwff_residual)
	output1 = permutedims(pwff_residual, [2,1,3])
    output2 = σ.(m.pwff_w_1(output1))
    #output3 = Flux.batched_transpose(m.pwff_w_2(output2))
	output3 = permutedims(m.pwff_w_2(output2), [2,1,3])
    output4 = m.dropout(output3)
    output5 = m.pwff_layer_norm(output4 .+ pwff_residual)
    return output5
end


# Global Self Attention
mutable struct Global_SelfAttn_Cell{A, B, C, D, E, F, G, H, I, J}
    conv::A
    dropout::B
    in_linear::C
    out_linear::D
    encoder_chain::E
    # Metadata
    inp::F
    d_model::G
    n_kernels::H
    window::I
    σ::J
end

Flux.@functor Global_SelfAttn_Cell
Flux.trainable(m::Global_SelfAttn_Cell) = (m.conv, m.in_linear, m.out_linear,
            m.encoder_chain)

"""
    Global_SelfAttn(inp, window, n_kernels, w_kernel, d_model, d_hid, n_layers, n_head)

Global self attention module for `DSANet`. For parameters see [DSANet](@ref).
"""
function Global_SelfAttn(inp, window, n_kernels, w_kernel, d_model, d_hid,
                n_layers, n_head, drop_prob = 0.1f0, σ = Flux.relu)

    conv = Conv((w_kernel, window), 1 => n_kernels)
    dropout = Dropout(drop_prob)
    in_linear = Dense(n_kernels, d_model)
    out_linear = Dense(d_model, n_kernels)
    encoder_chain = Chain([SelfAttn_Encoder(inp, d_model, n_head, d_hid, drop_prob, σ) for i=1:n_layers]...)
    return Global_SelfAttn_Cell(conv, dropout, in_linear, out_linear, encoder_chain,
                inp, d_model, n_kernels, window, σ)
end

function Base.show(io::IO, m::Global_SelfAttn_Cell)
	print(io, "Global_SelfAttn(", m.inp)
	print(io, ", ", m.window)
	print(io, ", ", m.n_kernels)
	print(io, ", ", size(m.conv.weight,1))
	print(io, ", ", m.d_model)
	print(io, ", ", m.encoder_chain[1].d_hid)
	print(io, ", ", length(m.encoder_chain))
	print(io, ", ", m.encoder_chain[1].n_head)
	m.dropout.p == 0.1 || print(io, ", ", m.dropout.p)
	m.σ == Flux.relu || print(io, ", ", m.σ)
	print(io, ")")
end

function (m::Global_SelfAttn_Cell)(x)
	x1 = m.conv(x)
    x2 = m.dropout(m.σ.(x1))
    x3 = dropdims(x2;dims=2)
    #x4 = Flux.batched_transpose(x3)
	x4 = permutedims(x3, [2,1,3])
    x5 = reshape(x4, (m.n_kernels, :))
    x6 = m.in_linear(x5)
    src_seq = reshape(x6, (m.d_model,m.inp,:))
    x8 = m.encoder_chain(src_seq)
    x9 = reshape(x8, (m.d_model, :))
    x10 = m.out_linear(x9)
    x11 = reshape(x10, (m.n_kernels, m.inp,:))
    return x11
end


# Local Self Attention
mutable struct Local_SelfAttn_Cell{A, B, C, D, E, F, G, H, I, J, K}
    conv::A
    dropout::B
    in_linear::C
    out_linear::D
    encoder_chain::E
    # Metadata
    inp::F
    d_model::G
    n_kernels::H
    window::I
    local_length::J
    σ::K
end

Flux.@functor Local_SelfAttn_Cell
Flux.trainable(m::Local_SelfAttn_Cell) = (m.conv, m.in_linear, m.out_linear,
                m.encoder_chain)

"""
    Local_SelfAttn(inp, window, local_length, n_kernels, w_kernel, d_model, d_hid, n_layers, n_head)

Local self attention module for `DSANet`. For parameters see [DSANet](@ref).
"""
function Local_SelfAttn(inp, window, local_length, n_kernels, w_kernel, d_model, d_hid,
                n_layers, n_head, drop_prob = 0.1f0, σ = Flux.relu)

    conv = Conv((w_kernel, local_length), 1 => n_kernels)
    dropout = Dropout(drop_prob)
    in_linear = Dense(n_kernels, d_model)
    out_linear = Dense(d_model, n_kernels)
    encoder_chain = Chain([SelfAttn_Encoder(inp, d_model, n_head, d_hid, drop_prob, σ) for i=1:n_layers]...)
    return Local_SelfAttn_Cell(conv, dropout, in_linear, out_linear, encoder_chain,
                inp, d_model, n_kernels, window, local_length, σ)
end

function Base.show(io::IO, m::Local_SelfAttn_Cell)
	print(io, "Local_SelfAttn(", m.inp)
	print(io, ", ", m.window)
	print(io, ", ", m.local_length)
	print(io, ", ", m.n_kernels)
	print(io, ", ", size(m.conv.weight,1))
	print(io, ", ", m.d_model)
	print(io, ", ", m.encoder_chain[1].d_hid)
	print(io, ", ", length(m.encoder_chain))
	print(io, ", ", m.encoder_chain[1].n_head)
	m.dropout.p == 0.1 || print(io, ", ", m.dropout.p)
	m.σ == Flux.relu || print(io, ", ", m.σ)
	print(io, ")")
end

function (m::Local_SelfAttn_Cell)(x)
	x1 = m.σ.(m.conv(x))
    x2 = MaxPool((1,m.window-m.local_length+1))(x1)
    x3 = m.dropout(x2)
    #x4 = Flux.batched_transpose(dropdims(x3, dims=(2)))
	x4 = permutedims(dropdims(x3, dims=(2)), [2,1,3])
    x5 = reshape(x4, (m.n_kernels, :))
    src_seq = reshape(m.in_linear(x5),(m.d_model, m.inp, :))
    x6 = m.encoder_chain(src_seq)
    x7 = reshape(x6, (m.d_model, :))
    x8 = m.out_linear(x7)
    x9 = reshape(x8, (m.n_kernels, m.inp,:))
    return x9
end


# Full DSANet
mutable struct DSANetCell{A, B, C, D, E, F, G, H, I, J}
    gsa::A
    lsa::B
    AR_linear::C
    output::D
	final_output::E
    dropout::F
    # Metadata
    inp::G
    n_kernels::H
    window::I
    output_dimensions::J
end

Flux.@functor DSANetCell
Flux.trainable(m::DSANetCell) = (m.gsa, m.lsa, m.AR_linear, m.output, m.final_output)

"""
    DSANet(inp, window, local_length, n_kernels, d_model, d_hid, n_layers, n_head, out=1, drop_prob = 0.1f0, σ = Flux.relu)

Creates a `DSANet` network based on the architecture described in
[Siteng Huang et. al.](https://kyonhuang.top/publication/dual-self-attention-network). The
code follows the [PyTorch implementation](https://github.com/bighuang624/DSANet).
`inp` specifies the number of input features. `window` gives the length of the window
for the pooled input data. `local_length` defines the length of the convolution window for
the local self attention mechanism. `n_kernel` defines the number of convolution kernels
for both the local and global self attention mechanism. `d_hid` defines the number of
"hidden" convolution kernels in the self attention encoder structure.
`n_layers` gives the number of self attention encoders used in the network, and `n_head`
defines the number of attention heads. `out` gives the number of output time series,
`drop_prob` is the dropout probability for the `Dropout` layers, and `σ` defines the
network's activation function.

Data is expected as array with dimensions `features x poolsize x 1 x data`, i.e.
for 1000 data points containing 31 features that have been windowed over 6
timesteps, `DSANet` expects an input size of `(31, 6, 1, 1000)`.
"""
function DSANet(inp, window, local_length, n_kernels, d_model, d_hid,
                n_layers, n_head, out=1, drop_prob = 0.1f0, σ = Flux.relu)
    out <= inp || throw(ArgumentError("Number of inputs ($inp) needs to be larger or equal to outputs ($out) "))
	w_kernel = 1
	gsa = Global_SelfAttn(inp, window, n_kernels, w_kernel, d_model, d_hid,
                    n_layers, n_head, drop_prob, σ)
    lsa = Local_SelfAttn(inp, window, local_length, n_kernels, w_kernel, d_model, d_hid,
                    n_layers, n_head, drop_prob, σ)
    AR_linear = Dense(window, 1)
    output = Dense(2*n_kernels, 1, σ)
	final_output = Dense(inp, out)
    dropout = Dropout(drop_prob)
    return DSANetCell(gsa, lsa, AR_linear, output, final_output, dropout, inp, n_kernels,
                    window, out)
end

function Base.show(io::IO, m::DSANetCell)
	print(io, "DSANet(", m.inp)
	print(io, ", ", m.window)
	print(io, ", ", m.lsa.local_length)
	print(io, ", ", m.lsa.n_kernels)
	#print(io, ", ", size(m.lsa.conv.weight,1))  # w_kernel
	print(io, ", ", m.lsa.d_model)
	print(io, ", ", m.lsa.encoder_chain[1].d_hid)
	print(io, ", ", length(m.lsa.encoder_chain))
	print(io, ", ", m.lsa.encoder_chain[1].n_head)
	m.output_dimensions == 1 || print(io, ", ", m.output_dimensions)
	m.dropout.p == 0.1f0 || print(io, ", ", m.dropout.p)
	m.lsa.σ == Flux.relu || print(io, ", ", m.lsa.σ)
	print(io, ")")
end

function (m::DSANetCell)(x)
	# Autoregressive Part
    x1 = dropdims(x, dims=3)
    #x2 = Flux.batched_transpose(x1)
	x2 = permutedims(x1, [2,1,3])
    x3 = reshape(x2, (m.window,:))
    x4 = m.AR_linear(x3)
    ar_output = reshape(x4,(m.inp,:))

    # Self Attention Modules
    gsa_output = m.gsa(x)
    lsa_output = m.lsa(x)

    # Full Net
    sf_output = cat(gsa_output, lsa_output, dims=1)
    sf_output1 = m.dropout(sf_output)
    sf_output2 = reshape(sf_output1, (2*m.n_kernels,:))
    sf_output3 = m.output(sf_output2)
    sf_output4 = reshape(sf_output3, (m.inp,:))

    out = sf_output4 .+ ar_output
    return m.final_output(out)
end
