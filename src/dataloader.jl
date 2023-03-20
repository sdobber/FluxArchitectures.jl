# Function for loading sample data

"""
    get_data(dataset, poollength, datalength, horizon)

Return features and labels from one of the sample datasets in the repository. `dataset`
can be one of `:solar`, `:traffic`, `:exchange_rate` or `:electricity`. `poollength` gives 
the number of timesteps to pool for the model, `datalength` determines the number of time steps
included into the output, and `horizon` determines the number of time steps that should be
forecasted by the model.

See also: [`prepare_data`](@ref), [`load_data`](@ref)
"""
get_data(dataset, poollength, datalength, horizon; normalise=true) =
  prepare_data(load_data(dataset), poollength, datalength, horizon; normalise=normalise)


"""
    prepare_data(data, poollength, datalength, horizon)
    prepare_data(data, poollength, datalength, horizon; normalise=true)

Cast 2D time series data into the format used by FluxArchitectures. `data` is a matrix or 
Tables.jl compatible datasource containing data in the form `timesteps x features` (i.e.
each column contains the time series for one feature). `poollength` defines the number of 
timesteps to pool when preparing a single frame of data to be fed to the model. `datalength` 
determines the number of time steps included into the output, and `horizon` determines the 
number of time steps that should be forecasted by the model. The label data is assumed to be 
contained in the first column. Outputs features and labels.

Note that when `horizon` is smaller or equal to `poollength`, then the model has direct
access to the value it is supposed to predict.
"""
function prepare_data(data::AbstractMatrix, poollength, datalength, horizon; normalise=true)
  extendedlength = datalength + poollength - 1
  extendedlength > size(data, 1) && throw(ArgumentError("datalength $(datalength) larger than available data $(size(data, 1) - poollength + 1)"))
  normalise && (data = Flux.normalise(data, dims=1))
  features = similar(data, size(data, 2), poollength, 1, datalength)
  for i in 1:datalength
    features[:, :, :, i] .= permutedims(data[i:(i+poollength-1), :])
  end
  labels = circshift(data[1:datalength, 1], -horizon)
  return features, labels
end

prepare_data(data, poollength, datalength, horizon; normalise=true) = prepare_data(Tables.matrix(data), poollength, datalength, horizon; normalise=normalise)


"""
    load_data(dataset)

Load the raw data from one of the available datasets.
The following example data from https://github.com/laiguokun/multivariate-time-series-data 
is included:

* `:solar`: The raw data is coming from http://www.nrel.gov/grid/solar-power-data.html: 
  It contains the solar power production records in the year of 2006, which is sampled every 
  10 minutes from 137 PV plants in Alabama State.

* `:traffic`: The raw data is coming from http://pems.dot.ca.gov. The data in this repo is a 
  collection of 48 months (2015-2016) hourly data from the California Department of 
  Transportation. The data describes the road occupancy rates (between 0 and 1) measured by 
  different sensors on San Francisco Bay area freeways.

* `:electricity`: The raw dataset is from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. 
  It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014 
  for 321 clients. The data has been cleaned and converted to hourly consumption.

* `:exchange_rate`: The collection of daily exchange rates of eight foreign countries 
  including Australia, Great Britain, Canada, Switzerland, China, Japan, New Zealand and 
  Singapore ranging from 1990 to 2016.
"""
function load_data(dataset)
  admissible = [:solar, :traffic, :exchange_rate, :electricity]
  dataset in admissible || throw(ArgumentError("Sample data $dataset not found"))

  datadir = joinpath(artifact"sample_data", "data")
  (dataset == :solar) && (BSON.@load joinpath(datadir, "solar_AL.bson") inp_raw)
  (dataset == :traffic) && (BSON.@load joinpath(datadir, "traffic.bson") inp_raw)
  (dataset == :exchange_rate) && (BSON.@load joinpath(datadir, "exchange_rate.bson") inp_raw)
  (dataset == :electricity) && (BSON.@load joinpath(datadir, "electricity.bson") inp_raw)
  return inp_raw
end