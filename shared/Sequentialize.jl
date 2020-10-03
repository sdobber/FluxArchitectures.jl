# Sequentialize.jl
#
# Sequentialization for recurrent networks
# The current Flux.jl setup does not allow to feed sequences as a matrix to a recurrent
# network. The constructions in this file help to unwrap a matrix to a sequence, apply
# an RNN and then wrap things up again.


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

HiddenRecur(m, h = Flux.hidden(m)) = HiddenRecur(m, h, h)

function (m::HiddenRecur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return collect(h)
end

Flux.@functor HiddenRecur cell, init

Base.show(io::IO, m::HiddenRecur) = print(io, "HiddenRecur(", m.cell, ")")
Flux.reset!(m::HiddenRecur) = (m.state = m.init)


"""
    Seq(RNN)

`Seq` takes a recurrent neural network "sequentializes" it, i.e. when `Seq(RNN)` is called
with a matrix of input features over a certain time interval, the recurrent neural net is
fed with a sequence of inputs, and results are transformed back to matrix form.
"""
mutable struct Seq{T}
    chain::T
    state
end

Seq(chain) = Seq(chain, [0.0f0])

# e.g. Seq(HiddenRecur(LSTMCell(10, 12))) returns hidden and cell state
# This variant is elegant but slow
# function (l::Seq)(x)
#     out = l.chain.(Flux.unstack(x,2))
#     if typeof(l.chain.state) <: Array || typeof(l.chain) <: Flux.Recur
#         l.state = Flux.stack(out,2)
#     elseif typeof(l.chain.state) <: Tuple
#         l.state = [Flux.stack(Flux.stack(out,1)[:,i],2) for i in 1:length(l.chain.state)]
#     end
#     return l.state
# end

# 1/3 of allocations, double as fast
function (l::Seq)(x)
    if typeof(l.chain.state) <: Array
        sizeval = (length(l.chain.state), size(x,2))
        out = Flux.Zygote.Buffer(x, sizeval)
        for i = 1:sizeval[2]
          out[:,i] = l.chain(x[:,i])
        end
        l.state = copy(out)
	elseif typeof(l.chain.state) <: Tuple && typeof(l.chain) <: Flux.Recur
        sizeval = (length(l.chain.state[1]), size(x,2))
        out = Flux.Zygote.Buffer(x, sizeval)
        for i = 1:sizeval[2]
          out[:,i] = l.chain(x[:,i])
        end
        l.state = copy(out)
    elseif typeof(l.chain.state) <: Tuple
        sizeval = (length(l.chain.init[1]), size(x,2))
        numhidden = length(l.chain.state)
        out = Flux.Zygote.Buffer(x, numhidden*sizeval[1], sizeval[2])
        for i=1:sizeval[2]
          out[:,i] = vcat(l.chain(x[:,i])...)
        end
        output = copy(out)
        l.state = [output[i*sizeval[1]+1:(i+1)*sizeval[1],:] for i=0:numhidden-1]
    end
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
Flux.trainable(m::Seq) = Flux.trainable(m.chain)


"""
    SeqSkip(RNNCell, skiplength::Integer)

`SeqSkip` takes a recurrent neural network cell and "sequentializes" it, i.e. when
it is called with a matrix of input features over a certain time interval, the recurrent
neural net is fed with a sequence of inputs, and results are transformed back to matrix
form. In addition, the hidden state from `skiplength` timesteps ago is used instead of the
current hidden state.

See also: [`Seq`](@ref)
"""
mutable struct SeqSkip{T}
    cell::T
	p  # skiplength
	init
	cellstate
	state
end

SeqSkip(m, p, h=Flux.hidden(m)) = SeqSkip(m, p, h, h, h)

function (l::SeqSkip)(x)
	if typeof(l.cellstate) <: Array
		sizeval = (length(l.init), size(x,2))
		out = Flux.Zygote.Buffer(x, sizeval)
		h, y = l.cell(l.cellstate, x[:,1])
		out[:,1] = h
		l.cellstate = h
		for i = 2:sizeval[2]
		  h, y = l.cell(out[:,max(1,i-l.p)], x[:,i])
		  l.cellstate = h
		  out[:,i] = h
		end
		l.state = copy(out)
	elseif typeof(l.cellstate) <: Tuple
		sizeval = (length(l.init[1]), size(x,2))
        numhidden = length(l.cellstate)
        out = Flux.Zygote.Buffer(x, numhidden*sizeval[1], sizeval[2])
		h, y = l.cell(l.cellstate, x[:,1])
		out[:,1] = vcat(h...)
		l.cellstate = h
		for i=2:sizeval[2]
	      hidden = Tuple([out[j*sizeval[1]+1:(j+1)*sizeval[1],max(1,i-l.p)] for j=0:numhidden-1 ])
		  h, y = l.cell(hidden, x[:,i])
		  l.cellstate = h
		  out[:,i] = vcat(h...)
        end
		output = copy(out)
		l.state = output[1:sizeval[1],:]
		# TO DO: For HiddenRecur-type this needs to be returning l.state
		# l.state = [output[i*sizeval[1]+1:(i+1)*sizeval[1],:] for i=0:numhidden-1]
	end
	return l.state
end

Flux.@functor SeqSkip cell, init

Base.show(io::IO, m::SeqSkip) = print(io, "SeqSkip(", m.cell, ")")
Flux.reset!(m::SeqSkip) = (m.cellstate = m.init)
