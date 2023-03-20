# Datasets

## Build-In

Currently, the following example datasets from [github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data) are included:

* `:solar`: The raw data is coming from [www.nrel.gov/grid/solar-power-data.html](http://www.nrel.gov/grid/solar-power-data.html): It contains the solar power production records in the year of 2006, which is sampled every 10 minutes from 137 PV plants in Alabama State.

* `:traffic`: The raw data is coming from [pems.dot.ca.gov](http://pems.dot.ca.gov). The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.

* `:electricity`: The raw dataset is from [archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014). It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014 for 321 clients. The data has been cleaned and converted to hourly consumption.

* `:exchange_rate`: The collection of daily exchange rates of eight foreign countries including Australia, Great Britain, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.


## Preparing your own data

The following example shows how to download a dataset from the web and extract features and labels from it using the [`prepare_data`](@ref)-function.

```julia
using CSVFiles
using DataFrames
using FluxArchitectures

src = "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
data = DataFrame(load(src))
target = :median_house_value
select!(data, Cols(target, :))  # make median_house_value the first column

poollength = 30
datalength = 5000
horizon = 10
features, labels = prepare_data(data, poollength, datalength, horizon; normalise=false)
```

Note that `prepare_data` expects the data that is supposed to be predicted in the first column, hence the sorting by `select!(data, Cols(target, :))`.