struct DummyState
    iteration::Int
    value::Float64
    g_norm::Float64
    metadata::Dict
end

@testset "Optimisation callback     " begin
    # construct callback cache
    t = OptimTrace([0.0])
    cb1 = ReSolver.CallbackCache(t, OptOptions(verbose=false))
    @test cb1 isa ReSolver.CallbackCache{<:OptimTrace{Nothing}, <:OptOptions}
    @test cb1.start_itr  == 0
    @test cb1.start_time == 0

    # test callback
    state1 = DummyState(0, 1, π, Dict("time"=>0, "Current step size"=>0.1, "x"=>randn(1)))
    state2 = DummyState(1, 0.8, π-2, Dict("time"=>0.2, "Current step size"=>0.05, "x"=>randn(1)))
    @test !cb1(state1)
    @test length(t) == 1
    @test t[1] == (0, 0.0, 0.1, 1.0, Float64(π), nothing)
    @test !cb1(state2)
    @test length(t) == 2
    @test t[2] == (1, 0.2, 0.05, 0.8, π-2, nothing)

    # construct new callback cache from previous trace
    cb2 = ReSolver.CallbackCache(t, OptOptions(verbose=false))
    @test cb2 isa ReSolver.CallbackCache{<:OptimTrace{Nothing}, <:OptOptions}
    @test cb2.start_itr == 1
    @test cb2.start_time == 0.2
    state3 = DummyState(0, 0.8, π-2, Dict("time"=>0, "Current step size"=>0.05, "x"=>randn(1)))
    state4 = DummyState(1, 0.4, π-3.1, Dict("time"=>0.5, "Current step size"=>0.03, "x"=>randn(1)))
    @test !cb2(state3)
    @test length(t) == 3
    @test t[3] == (1, 0.2, 0.05, 0.8, π-2, nothing)
    @test !cb2(state4)
    @test length(t) == 4
    @test t[4] == (2, 0.7, 0.03, 0.4, π-3.1, nothing)
end
