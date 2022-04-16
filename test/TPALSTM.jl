@testset "shapes" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = TPALSTM(inputsize, 10, poollength, 2, 32, 1)
    @test size(m(x)) == (1, datalength)
    if Flux.CUDA.functional()
        @test size(gpu(m)(gpu(x))) == (1, datalength)
    end
    FluxArchitectures.initialize_bias!(m)
    @test all([all(m.lstm.chain.chain[i].cell.b .== 1) for i in 1:length(m.lstm.chain.chain)])
end

@testset "misc" begin
    @test repr(TPALSTM(10, 10, 10, 2, 32, 1)) == "TPALSTM(10, 10, 10, 2)"
    @test repr(TPALSTM(10, 10, 10, 2, 20, 4)) == "TPALSTM(10, 10, 10, 2, 20, 4)"
end

@testset "gradients" begin
    inputsize = 20
    poollength = 10
    datalength = 100
    x = rand(Float32, inputsize, poollength, 1, datalength)
    m = TPALSTM(inputsize, 10, poollength, 2, 32, 1)
    @test_nothrow fw_cpu(m, x)
    @test_nothrow bw_cpu(m, x)
    if Flux.CUDA.functional()
        @test_nothrow fw_gpu(m, x)
        @test_nothrow bw_gpu(m, x)
    end
end