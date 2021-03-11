# Sequentialize.jl
#
# Sequentialization for recurrent networks
# The current Flux.jl setup does not allow to feed sequences as a matrix to a recurrent
# network. The constructions in this file help to unwrap a matrix to a sequence, apply
# an RNN and then wrap things up again.
include("StackedLSTM.jl")

"""
    HiddenRecur(cell)

`HiddenRecur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. As opposed to `Recur`, it returns the both the hidden state and the
cell state.

See also: [`Recur`](@ref)
"""
mutable struct HiddenRecur{T}
    cell::T
    init
    state
end

HiddenRecur(m, h=Flux.hidden(m)) = HiddenRecur(m, h, h)

function (m::HiddenRecur)(xs...)
    h, y = m.cell(m.state, xs...)
    m.state = h
    return h  # collect(h)
end

Flux.@functor HiddenRecur cell, init

Base.show(io::IO, m::HiddenRecur) = print(io, "HiddenRecur(", m.cell, ")")
Flux.reset!(m::HiddenRecur) = (m.state = m.init)
Flux.trainable(m::HiddenRecur) = (m.cell,)


"""
    Seq(RNN)

`Seq` takes a recurrent neural network and "sequentializes" it, i.e. when `Seq(RNN)` is
called with a matrix of input features over a certain time interval, the recurrent neural
net is fed with a sequence of inputs, and results are transformed back to matrix form.
"""
mutable struct Seq{T}
    chain::T
    state
end

Seq(chain) = Seq(chain, [0.0f0])
(l::Seq)(x) = l(l.chain.state, l.chain, x)
getbuffersize(chain, x) = getbuffersize(typeof(chain), chain.state, x)

# CPU Arrays
function (l::Seq)(::Array, _, x)
	l.state = Align(map(l.chain, Slices(x, True(), False())), 1)
	return l.state
end
function (l::Seq)(::Array, ::Flux.Recur, x)
	l.state = Align(map(l.chain, Slices(x, True(), False())), 1)
	return l.state
end
function (l::Seq)(::Tuple, ::Flux.Recur, x)
	l.state = Align(map(l.chain, Slices(x, True(), False())), 1)
	return l.state
end
function (l::Seq)(::Tuple, _, x)
	tuples = map(l.chain, Slices(x, True(), False()))
	l.state = [Align(map(x -> x[i], tuples), 1) for i in 1:length(l.chain.state)]
	return l.state
end

# GPU Arrays
getbuffersize(::Type{<:Union{Flux.Recur,StackedLSTMCell}},
 				state::AbstractArray, x) = ((length(state), size(x, 2)), 1)
getbuffersize(::Type{<:Union{Flux.Recur,StackedLSTMCell}},
				state::Tuple, x) = ((length(state[1]), size(x, 2)), 1)
getbuffersize(::Type{<:Union{HiddenRecur}},
				state::Tuple, x) = ((length(state[1]), size(x, 2)), length(state))

function writebuffer(chain::HiddenRecur, x)
	sizeval, numhidden = getbuffersize(chain, x)
	out = Flux.Zygote.Buffer(x, numhidden * sizeval[1], sizeval[2])
	for i = 1:sizeval[2]
	  out[:,i] = cat(chain(x[:,i])...; dims=1)
	end
	output = copy(out)
	return [output[i * sizeval[1] + 1:(i + 1) * sizeval[1],:] for i = 0:numhidden - 1]
end

function writebuffer(chain, x)
	sizeval, numhidden = getbuffersize(chain, x)
	out = Flux.Zygote.Buffer(x, sizeval)
	for i = 1:sizeval[2]
	  out[:,i] = chain(x[:,i])
	end
	return copy(out)
end

function (l::Seq)(::Flux.CUDA.CuArray, _, x)
	l.state = writebuffer(l.chain, x)
	return l.state
end

function (l::Seq)(::Tuple{Flux.CUDA.CuArray, Flux.CUDA.CuArray}, _, x)
	l.state = writebuffer(l.chain, x)
	return l.state
end

function Base.show(io::IO, l::Seq)
    print(io, "Seq(", l.chain)
    print(io, ")")
end

function Flux.reset!(m::Seq)
    Flux.reset!(m.chain)
    m.state = [0.0f0]
    return nothing
end

Flux.@functor Seq
Flux.trainable(m::Seq) = Flux.trainable(m.chain)


"""
    SeqSkip(RNNCell, skiplength::Integer)

`SeqSkip` takes a recurrent neural network cell and "sequentializes" it, i.e. when
it is called with a matrix of input features over a certain time interval, the recurrent
neural net is fed with a sequence of inputs, and results are transformed back to matrix
form. In addition, the hidden state from `skiplength` timesteps ago is used instead of the
current hidden state. This structure combines functionality of `Recur` in that it makes a
recurrent neural network cell stateful, as well as `Seq` in that it feeds matrices of
input features as elements of a time series.

See also: [`Seq`](@ref), [Recur](@ref)
"""
mutable struct SeqSkip{T}
	cell::T
	p  # skiplength
	init
	state  # current cell state
	fullstate  # cell states over time series
end

SeqSkip(m, p, h=Flux.hidden(m)) = SeqSkip(m, p, h, h, h)

getbuffersize(::Type{<:SeqSkip}, state::AbstractArray, x) = ((length(state), size(x, 2)), 1)
getbuffersize(::Type{<:SeqSkip}, state::Tuple, x) = ((length(state[1]), size(x, 2)), length(state))

writebuffer(l::SeqSkip,x) = writebuffer(typeof(l.state), l, x)

function writebuffer(::Type{<:Union{AbstractArray}}, l::SeqSkip, x)
	sizeval, numhidden = getbuffersize(l, x)
	out = Flux.Zygote.Buffer(x, sizeval)
	h, y = l.cell(l.state, x[:,1])
	out[:,1] = h
	l.state = h
	for i = 2:sizeval[2]
	  h, y = l.cell(out[:,max(1, i - l.p)], x[:,i])
	  l.state = h
	  out[:,i] = h
	end
	l.fullstate = copy(out)
	return l.fullstate
end

function writebuffer(::Type{<:Tuple}, l::SeqSkip, x)
	sizeval, numhidden = getbuffersize(l, x)
	out = Flux.Zygote.Buffer(x, numhidden * sizeval[1], sizeval[2])
	h, y = l.cell(l.state, x[:,1])
	out[:,1] = vcat(h...)
	l.state = h
	for i = 2:sizeval[2]
		hidden = Tuple([out[j * sizeval[1] + 1:(j + 1) * sizeval[1],max(1, i - l.p)] for j = 0:numhidden - 1 ])
		h, y = l.cell(hidden, x[:,i])
		l.state = h
		out[:,i] = vcat(h...)
	end
	l.fullstate = copy(out)
	return l.fullstate[1:sizeval[1],:]
end
# TO DO: For HiddenRecur-type we need writebuffer that outputs `output`

(l::SeqSkip)(x) = writebuffer(l, x)

Flux.@functor SeqSkip

Base.show(io::IO, m::SeqSkip) = print(io, "SeqSkip(", m.cell, ")")
Flux.reset!(m::SeqSkip) = (m.state = m.init)
