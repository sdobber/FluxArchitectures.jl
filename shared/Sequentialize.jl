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
function (l::Seq)(x)
    out = l.chain.(Flux.unstack(x,2))
    if typeof(l.chain.state) <: Array || typeof(l.chain) <: Flux.Recur
        l.state = Flux.stack(out,2)
    elseif typeof(l.chain.state) <: Tuple
        l.state = [Flux.stack(Flux.stack(out,1)[:,i],2) for i in 1:length(l.chain.state)]
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
