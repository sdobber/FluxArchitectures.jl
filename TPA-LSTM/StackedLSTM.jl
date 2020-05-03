# StackedLSTM.jl
#
# Layers for stacked LSTM

mutable struct StackedLSTMCell{A}
	chain::A
end

"""
    HiddenRecur(cell)

`HiddenRecur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. As opposed to `Recur`, it returns the hidden state from the cell when
called, and not the cell output.

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
  return h[1]
end

Flux.@functor HiddenRecur cell, init

Base.show(io::IO, m::HiddenRecur) = print(io, "HiddenRecur(", m.cell, ")")

Flux.trainable(m::StackedLSTMCell) = (m.chain)
Flux.reset!(m::HiddenRecur) = (m.state = m.init)

"""
	StackedLSTM(in, out, hiddensize, layers)

Stacked LSTM network. Feeds the data through a chain of LSTM layers, where the hidden state
of the previous layer gets fed to the next one. The first layer corresponds to
`LSTM(in, hiddensize)`, the hidden layers to `LSTM(hiddensize, hiddensize)`, and the final
layer to `LSTM(hiddensize, out)`. Takes the keyword argument `init` for the initialization
of the layers.
"""
function StackedLSTM(in::Integer, out::Integer, hiddensize::Integer, layers::Integer;
			init = Flux.glorot_uniform)
	if layers == 1
		chain = Chain(LSTM(in, out; init = init))
	elseif layers == 2
		chain = Chain(HiddenRecur(Flux.LSTMCell(in, hiddensize; init = init)),
					LSTM(hiddensize, out; init = init))
	else
		chain_vec=[HiddenRecur(Flux.LSTMCell(in, hiddensize; init = init))]
		for i=1:layers-2
			push!(chain_vec, HiddenRecur(Flux.LSTMCell(hiddensize, hiddensize; init = init)))
		end
		chain = Chain(chain_vec..., LSTM(hiddensize, out; init = init))
	end
	return StackedLSTMCell(chain)
end

function (m::StackedLSTMCell)(x)
	return m.chain(x)
end

function Base.show(io::IO, l::StackedLSTMCell)
	if length(l.chain)==1
		print(io, l.chain[1])
	else
		print(io, "StackedLSTMCell(", size(l.chain[1].cell.Wi, 2), ", ", size(l.chain[end].cell.Wi, 1)÷4)
		print(io, ", ", size(l.chain[end].cell.Wi, 2))
		print(io, ", ", length(l.chain), ")")
	end
end
