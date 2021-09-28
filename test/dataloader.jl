@testset "size" begin
    poollength = 10
    horizon = 6
    datalength = 500
    input, target = get_data(:solar, poollength, datalength, horizon)
    @test size(input) == (137, poollength, 1, datalength)
    @test size(target) == (500,)
    @test_throws ArgumentError get_data(:solar, poollength, 100000, horizon)
end 

@testset "data" begin
    poollength = 10
    horizon = 6
    datalength = 500
    for dataset in (:solar, :traffic, :exchange_rate, :electricity)
        input, target = get_data(dataset, poollength, datalength, horizon)
        @test size(input, 1) >= 1
    end
    @test_throws ArgumentError get_data(:mnist, poollength, datalength, horizon)
end

@testset "memory issues" begin
    data = ones(Float32, 100, 10)
    input, target = FluxArchitectures.prepare_data(data, 6, 70, 10, normalise=false)
    @test all(input .== 1.0)
    @test all(target .== 1.0)
end

