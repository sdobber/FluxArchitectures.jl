@testset "StackedLSTM" begin
    x = rand(Float32, 20, 1, 1)
    for m in (FluxArchitectures.StackedLSTM(20, 10, 3, 1), FluxArchitectures.StackedLSTM(20, 10, 3, 2),
        FluxArchitectures.StackedLSTM(20, 10, 3, 5))
        @test size(m(x)) == (10, 1, 1)
    end
    @test repr(FluxArchitectures.StackedLSTM(20, 10, 3, 1)) == "Recur(LSTMCell(20, 10))"
    @test repr(FluxArchitectures.StackedLSTM(20, 10, 3, 2)) == "StackedLSTM(20, 10, 3, 2)"
    m = FluxArchitectures.StackedLSTM(20, 10, 3, 5)
    FluxArchitectures.initialize_bias!(m)
    @test all([all(m.chain[i].cell.b .== 1) for i in 1:length(m.chain)])
end 

@testset "HiddenRecur" begin
    x = rand(Float32, 20, 100)
    m = FluxArchitectures.HiddenRecur(Flux.LSTMCell(20, 10))
    @test size(m(x)[1]) == (10, 100)
    @test size(m(x)[2]) == (10, 100)
    m = FluxArchitectures.HiddenRecur(Flux.GRUCell(20, 10))
    @test size(m(x)) == (10, 100)
    @test repr(m) == "HiddenRecur(GRUCell(20, 10))"
end

@testset "Sequentialize" begin
    x = rand(Float32, 20, 100)
    m = FluxArchitectures.Seq(LSTM(20, 10))
    @test repr(m) == "Seq(Recur(LSTMCell(20, 10)))"
    @test size(m(x)) == (10, 100)
    m = FluxArchitectures.Seq(FluxArchitectures.HiddenRecur(Flux.LSTMCell(20, 10)))
    @test size(m(x)[1]) == (10, 100)
    @test size(m(x)[2]) == (10, 100)
    m = FluxArchitectures.SeqSkip(Flux.LSTMCell(20, 10), 10)
    @test repr(m) == "SeqSkip(LSTMCell(20, 10))" 
    @test size(m(x)) == (10, 100)
    m = FluxArchitectures.SeqSkip(Flux.GRUCell(20, 10), 10)
    @test size(m(x)) == (10, 100)
end