module FluxArchitectures

import Flux: reset!, trainable

using Reexport
@reexport using Flux

using BSON
using LazyArtifacts

import Tables

include("./shared/StackedLSTM.jl")
include("./shared/Sequentialize.jl")
include("dataloader.jl")
include("DARNN.jl")
include("DSANet.jl")
include("LSTnet.jl")
include("TPALSTM.jl")

export DARNN, DSANet, LSTnet, TPALSTM, get_data, prepare_data

end # module
