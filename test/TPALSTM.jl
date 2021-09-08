@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = TPALSTM(inputsize, 10, poollength, 2, 32, 1)
    @test size(m(x)) == (1, datalength)
end 

@testset "misc" begin
    @test repr(TPALSTM(10, 10, 10, 2, 32, 1)) == "TPALSTM(10, 10, 10, 2)"
    @test repr(TPALSTM(10, 10, 10, 2, 20, 4)) == "TPALSTM(10, 10, 10, 2, 20, 4)"
end