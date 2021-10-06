@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = DARNN(inputsize, 10, 10, poollength, 1)
    @test size(m(x)) == (1, datalength)
    if Flux.CUDA.functional()
        @test size(gpu(m)(gpu(x))) == (1, datalength)
    end
end 

@testset "misc" begin
    @test repr(DARNN(10, 10, 10, 1, 1)) == "DARNN(10, 10, 10, 1, 1)"
end

# @testset "constructors" begin
#     @test size(Dense(10, 100).weight) == (100, 10)
#     @test size(Dense(10, 100).bias) == (100,)
#     @test Dense(rand(100, 10), rand(100)).σ == identity
#     @test Dense(rand(100, 10)).σ == identity
# end