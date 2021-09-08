@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = DSANet(inputsize, poollength, 3, 3, 4, 1, 3, 2)
    @test size(m(x)) == (1, datalength)
end 

@testset "misc" begin
    @test repr(DSANet(10, 10, 3, 3, 4, 1, 3, 2)) == "DSANet(10, 10, 3, 3, 4, 1, 3, 2)"
end