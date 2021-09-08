@testset "size" begin
    poollength = 10
    horizon = 6
    datalength = 500
    input, target = get_data(:solar, poollength, datalength, horizon)
    @test size(input) == (137, poollength, 1, datalength)
    @test size(target) == (500,)
end 

@testset "data" begin
    poollength = 10
    horizon = 6
    datalength = 500
    for dataset in (:solar, :traffic, :exchange_rate, :electricity)
        input, target = get_data(dataset, poollength, datalength, horizon)
        @test size(input, 1) >= 1
    end
    @test_throws ErrorException get_data(:mnist, poollength, datalength, horizon)
end