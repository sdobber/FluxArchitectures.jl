```@meta
CurrentModule = FluxArchitectures
```

# Examples

## LSTnet - Copy-Paste Code

If you want to start right away, make sure that `FluxArchitectures` and `Plots` are installed, and try the following. Details are below.
```julia
using FluxArchitectures, Plots

@info "Loading data"
poollength = 10
horizon = 6
datalength = 1000
input, target = get_data(:exchange_rate, poollength, datalength, horizon) |> gpu

@info "Creating model and loss"
inputsize = size(input, 1)
convlayersize = 2
recurlayersize = 3
skiplength = 120
model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
        init=Flux.zeros32, initW=Flux.zeros32) |> gpu

function loss(x, y)
    Flux.reset!(model)
    return Flux.mse(model(x), y')
end

cb = function ()
    Flux.reset!(model)
    pred = model(input)' |> cpu
    Flux.reset!(model)
    p1 = plot(pred, label="Predict")
    p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
    display(plot(p1))
end

@info "Start loss" loss = loss(input, target)
@info "Starting training"
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 20),
            ADAM(0.01), cb=cb)
@info "Final loss" loss = loss(input, target)
```


## LSTnet - Step-by-step

### Load some sample data

We start out by loading some of the example [Datasets](@ref) - in this case the `:exchange_rate` dataset, a collection of daily exchange rates of eight foreign countries. To speed up training, we only take the first 1000 time steps from the data. We would like to feed the model with a window of 10 past timesteps while at the same time trying to forecast 6 timesteps in the future.
```julia
using FluxArchitectures, Plots

poollength = 10
horizon = 6
datalength = 1000
input, target = get_data(:exchange_rate, poollength, datalength, horizon) |> gpu
```


### Define the neural net and loss

We use a `LSTnet` model with 2 convolutional layers, and 3 recurrent layers. For the recurrent-skip component, we use a hidden state 120 time steps from the past.
```julia
inputsize = size(input, 1)
convlayersize = 2
recurlayersize = 3
skiplength = 120
model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength,
        init=Flux.zeros32, initW=Flux.zeros32) |> gpu
```

As the loss function, we use the standard mean squared error loss. To make sure to reset the hidden state for each training loop, we call `Flux.reset!` every time we calculate the loss.
```julia
function loss(x, y)
    Flux.reset!(model)
    return Flux.mse(model(x), y')
end
```


### Callback for plotting the training

To observe the training progress, we define the following function to plot the training data and prediction together with the current loss value.
```julia
cb = function ()
    Flux.reset!(model)
    pred = model(input)' |> cpu
    Flux.reset!(model)
    p1 = plot(pred, label="Predict")
    p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
    display(plot(p1))
end
```

### Training loop

Finally, we start the training loop and train for 20 epochs.
```julia
@info "Start loss" loss = loss(input, target)
@info "Starting training"
Flux.train!(loss, Flux.params(model),Iterators.repeated((input, target), 20),
            ADAM(0.01), cb=cb)
@info "Final loss" loss = loss(input, target)
```