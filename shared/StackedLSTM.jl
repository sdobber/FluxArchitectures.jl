# StackedLSTM.jl
#
# Layers for stacked LSTM

mutable struct StackedLSTMCell{A}
	chain::A
	state
end

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
		chain = Chain(LSTM(in, hiddensize; init = init),
					  LSTM(hiddensize, out; init = init))
	else
		chain_vec=[LSTM(in, hiddensize; init = init)]
		for i=1:layers-2
			push!(chain_vec, LSTM(hiddensize, hiddensize; init = init))
		end
		chain = Chain(chain_vec..., LSTM(hiddensize, out; init = init))
	end
	return StackedLSTMCell(chain,  zeros(Float32,out))
end

function (m::StackedLSTMCell)(x)
	out = m.chain(x)
	m.state = out
	return out
end

Flux.@functor StackedLSTMCell 
Flux.trainable(m::StackedLSTMCell) = (m.chain)

function Base.show(io::IO, l::StackedLSTMCell)
	if length(l.chain)==1
		print(io, l.chain[1])
	else
		print(io, "StackedLSTMCell(", size(l.chain[1].cell.Wi, 2), ", ", size(l.chain[end].cell.Wi, 1)÷4)
		print(io, ", ", size(l.chain[end].cell.Wi, 2))
		print(io, ", ", length(l.chain), ")")
	end
end

# Initialize forget gate bias with 1
function initialize_bias!(l::StackedLSTMCell)
	for i=1:length(l.chain)
		l.chain[i].cell.b .= 1
	end
	return nothing
end
