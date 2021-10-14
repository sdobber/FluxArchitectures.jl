@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = LSTnet(inputsize, 2, 3, poollength, 20)
    @test size(m(x)) == (1, datalength)
    if Flux.CUDA.functional()
        @test size(gpu(m)(gpu(x))) == (1, datalength)
    end
    FluxArchitectures.initialize_bias!(m)
	@test all(m.RecurLayer.chain.cell.b .== 1)
	@test all(m.RecurSkipLayer.cell.b .== 1)
end 

@testset "ReluGRU" begin
    inputsize = 20
    datalength = 100
    x = rand(Float32, inputsize, datalength)
    m = FluxArchitectures.ReluGRU(inputsize, 10)
    @test size(m(x)) == (10, datalength)
    if Flux.CUDA.functional()
        @test size(gpu(m)(gpu(x))) == (10, datalength)
    end
end

@testset "misc" begin
    @test repr(LSTnet(10, 2, 3, 10, 20)) == "LSTnet(10, 2, 3, 10, 20)"
    @test repr(FluxArchitectures.ReluGRU(20, 10)) == "Recur(ReluGRUCell(20, 10))"
end

@testset "trainable" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = LSTnet(inputsize, 2, 3, poollength, 20)
    @test @no_error bw_cpu(m, x)
    if Flux.CUDA.functional()
        @test @no_error bw_gpu(m, x)
    end
end