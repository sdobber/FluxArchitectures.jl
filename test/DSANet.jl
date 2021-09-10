@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = DSANet(inputsize, poollength, 3, 3, 4, 1, 3, 2)
    @test size(m(x)) == (1, datalength)
    if Flux.use_cuda[]
        @test size(gpu(m)(gpu(x))) == (1, datalength)
    end
end 

@testset "misc" begin
    @test repr(DSANet(10, 10, 3, 3, 4, 1, 3, 2)) == "DSANet(10, 10, 3, 3, 4, 1, 3, 2)"
    @test repr(FluxArchitectures.Global_SelfAttn(10, 10, 3, 3, 4, 1, 3, 2)) == "Global_SelfAttn(10, 10, 3, 3, 4, 1, 3, 2, 0.1)"
    @test repr(FluxArchitectures.Local_SelfAttn(10, 10, 3, 3, 4, 4, 3, 3, 2)) == "Local_SelfAttn(10, 10, 3, 3, 4, 4, 3, 3, 2, 0.1)"
    @test_throws ArgumentError FluxArchitectures.Local_SelfAttn(10, 10, 3, 3, 4, 1, 3, 3, 2)
    @test_throws ArgumentError DSANet(10, 10, 3, 3, 4, 1, 3, 2, 20)
end