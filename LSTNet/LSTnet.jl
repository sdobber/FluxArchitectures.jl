# LSTNet.jl
#
# Layers for LSTNet


# Relu-GRU

mutable struct ReluGRUCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
end

ReluGRUCell(in, out; init = Flux.glorot_uniform) =
  ReluGRUCell(init(out*3, in), init(out*3, out),
          init(out*3), zeros(Float32, out))

function (m::ReluGRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wi*x, m.Wh*h
  r = σ.(Flux.gate(gx, o, 1) .+ Flux.gate(gh, o, 1) .+ Flux.gate(b, o, 1))
  z = σ.(Flux.gate(gx, o, 2) .+ Flux.gate(gh, o, 2) .+ Flux.gate(b, o, 2))
  h̃ = relu.(Flux.gate(gx, o, 3) .+ r .* Flux.gate(gh, o, 3) .+ Flux.gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

Flux.hidden(m::ReluGRUCell) = m.h

Flux.@functor ReluGRUCell

Base.show(io::IO, l::ReluGRUCell) =
  print(io, "ReluGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    ReluGRU(in::Integer, out::Integer)

Gated Recurrent Unit layer with `relu` as activation function.
"""
ReluGRU(a...; ka...) = Flux.Recur(ReluGRUCell(a...; ka...))




# Skip-GRU

skipgate(h, n, p) = (1:h) .+ h*(n-1)
skipgate(x::AbstractVector, h, n, p) = x[skipgate(h,n,p)]
skipgate(x::AbstractMatrix, h, n, p) = x[skipgate(h,n,p),circshift(1:size(x,2),-p)]

mutable struct SkipGRUCell{N,A,V}
  p::N
  Wi::A
  Wh::A
  b::V
  h::V
end

SkipGRUCell(in, out, p; init = Flux.glorot_uniform) =
  SkipGRUCell(p, init(out*3, in), init(out*3, out),
          init(out*3), zeros(Float32, out))

function (m::SkipGRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wi*x, m.Wh*h
  p = m.p
  r = σ.(Flux.gate(gx, o, 1) .+ skipgate(gh, o, 1, p) .+ Flux.gate(b, o, 1))
  z = σ.(Flux.gate(gx, o, 2) .+ skipgate(gh, o, 2, p) .+ Flux.gate(b, o, 2))
  h̃ = relu.(Flux.gate(gx, o, 3) .+ r .* skipgate(gh, o, 3, p) .+ Flux.gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

Flux.hidden(m::SkipGRUCell) = m.h

Flux.@functor SkipGRUCell

Base.show(io::IO, l::SkipGRUCell) =
  print(io, "SkipGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ", ", l.p  , ")")

"""
    SkipGRU(in::Integer, out::Integer, p::Integer)

Skip Gated Recurrent Unit layer with skip length `p`. The hidden state is recalled
from `p` steps prior to the current calculation.
"""
SkipGRU(a...; ka...) = Flux.Recur(SkipGRUCell(a...; ka...))


# LSTnet
mutable struct LSTnetCell{A, B, C, D, G}
  ConvLayer::A
  RecurLayer::B
  RecurSkipLayer::C
  RecurDense::D
  AutoregLayer::G
end

"""
    LSTnet(in, convlayersize, recurlayersize, poolsize, skiplength)
    LSTnet(in, convlayersize, recurlayersize, poolsize, skiplength, Flux.relu)

Creates a LSTnet layer based on the architecture described in
[Lai et. al.](https://arxiv.org/abs/1703.07015). `in` specifies the Number of
input features. `convlayersize` defines the number of convolutional layers, and
`recurlayersize` defines the number of recurrent layers. `poolsize` gives the
length of the window for the pooled input data, and `skiplength` defines the
number of steps the hidden state of the recurrent layer is taken back in time.

Data is expected as array with dimensions `features x poolsize x 1 x data`, i.e.
for 1000 data points containing 31 features that have been windowed over 6
timesteps, `LSTNet` expects an input size of `(31, 6, 1, 1000)`.

Takes the keyword arguments `init` for the initialization of the recurrent
layers; and `initW` and `initb` for the initialization of the dense layer.
"""
function LSTnet(in::Integer, convlayersize::Integer, recurlayersize::Integer, poolsize::Integer, skiplength::Integer, σ = Flux.relu;
	init = Flux.glorot_uniform, initW = Flux.glorot_uniform, initb = Flux.zeros)

	CL = Chain(Conv((in, poolsize), 1 => convlayersize, σ),
			a -> dropdims(a, dims = (1,2)))
	RL = Seq(ReluGRU(convlayersize,recurlayersize; init = init))
	RSL = Seq(SkipGRU(convlayersize,recurlayersize, skiplength; init = init))
	RD = Chain(Dense(2*recurlayersize, 1, identity))
	AL = Chain(a -> a[:,end,1,:], Dense(in, 1 , identity; initW = initW, initb = initb) )

    LSTnetCell(CL, RL, RSL, RD, AL)
end

function (m::LSTnetCell)(x)
	modelRL1 = m.RecurLayer(m.ConvLayer(x))
	modelRL2 = m.RecurSkipLayer(m.ConvLayer(x))
	modelRL =  m.RecurDense(cat(modelRL1, modelRL2; dims=1))
	return modelRL + m.AutoregLayer(x)
end

function Base.show(io::IO, l::LSTnetCell)
  print(io, "LSTnet(", size(l.ConvLayer[1].weight,1))
  print(io, ", ", size(l.RecurLayer.chain.cell.Wi, 2), ", ", size(l.RecurLayer.chain.cell.Wi, 1)÷3 )
  print(io, ", ", size(l.ConvLayer[1].weight,2))
  print(io, ", ", l.RecurSkipLayer.chain.cell.p)
  l.ConvLayer[1].σ == Flux.relu || print(io, ", ", l.ConvLayer[1].σ)
  print(io, ")")
end

Flux.params(m::LSTnetCell) = Flux.params(m.ConvLayer, m.RecurLayer, m.RecurSkipLayer, m.RecurDense, m.AutoregLayer)
Flux.reset!(m::LSTnetCell) = Flux.reset!.((m.ConvLayer, m.RecurLayer, m.RecurSkipLayer, m.RecurDense, m.AutoregLayer))
