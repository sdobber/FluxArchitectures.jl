@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = LSTnet(inputsize, 2, 3, poollength, 20)
    @test size(m(x)) == (1, datalength)
end 

@testset "misc" begin
    @test repr(LSTnet(10, 2, 3, 10, 20)) == "LSTnet(10, 2, 3, 10, 20)"
end