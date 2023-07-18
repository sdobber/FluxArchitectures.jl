using FluxArchitectures

@info "Loading data"
poollength = 10
horizon = 15
datalength = 1000
input, target = get_data(:exchange_rate, poollength, datalength, horizon) |> gpu

@info "Creating model and loss"
inputsize = size(input, 1)
convlayersize = 2
recurlayersize = 3
skiplength = 120
model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength, init=Flux.zeros32, initW=Flux.zeros32) |> gpu

function loss(x, y)
    Flux.reset!(model)
    return Flux.mse(model(x), y')
end

cb = function ()
    Flux.reset!(model)
    pred = model(input)' |> cpu
    Flux.reset!(model)
    #p1 = plot(pred, label="Predict")
    #p1 = plot!(cpu(target), label="Data", title="Loss $(loss(input, target))")
    #display(plot(p1))
end

@info "Start loss" loss = loss(input, target)
@info "Starting training"
Flux.train!(loss, Flux.params(model), Iterators.repeated((input, target), 20), Adam(0.01), cb=cb)
@info "Final loss" loss = loss(input, target)


# -----------------

using FluxArchitectures

function bw_cpu(m, ip)
    fun = x -> m(x)
    gs = gradient((x) -> sum(fun(x)), ip)
end

inputsize = 20
poollength = 10
datalength = 100
x = rand(Float32, inputsize, poollength, 1, datalength)
m = DARNN(inputsize, 10, 10, poollength, 1)
bw_cpu(m, x)


# error happens between Zyote v0.6.43 -> 0.6.44

# -------

using Flux

m = LSTM(20 => 1)
x = rand(Float32, 20, datalength)
m(x)
bw_cpu(m, x)
