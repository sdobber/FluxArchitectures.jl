module FluxArchitectures

import Flux:reset!

using Reexport
@reexport using Flux

using BSON
using JuliennedArrays
using SliceMap

include("./shared/StackedLSTM.jl")
include("./shared/Sequentialize.jl")
include("dataloader.jl")
include("DARNN.jl")
include("DSANet.jl")
include("LSTnet.jl")
include("TPALSTM.jl")

export get_data, DARNN, DSANet, LSTnet, TPALSTM

end # module
