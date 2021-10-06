# Exported Functions

```@meta
CurrentModule = FluxArchitectures
```

## Data Preparation

When raw data is available in matrix format, then `prepare_data` allows for easy conversion and cropping to the format expected by the models. It also accepts inputs compatible to the Tables.jl interface, for example a `DataFrame` or `CSV.File`.
```@docs
prepare_data
```

For loading some example data, the following function can be used.
```@docs
get_data
```
The datasets are automatically downloaded when needed. See [Datasets](@ref) for a description.


## Models

The following models are exported:
```@docs
DARNN
DSANet
LSTnet
TPALSTM
```

