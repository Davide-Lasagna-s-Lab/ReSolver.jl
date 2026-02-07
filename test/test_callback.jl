struct DummyState
    iteration::Int
    value::Float64
    g_norm::Float64
    metadata::Dict
end

@testset "Optimisation callback     " begin
    # construct callback cache
    s = OptState([0.0])
    cb1 = ReSolver.CallbackCache(s, OptOptions(verbose=false))
    @test cb1 isa ReSolver.CallbackCache{<:OptState{Vector{Float64}, Float64, Nothing}, <:OptOptions}
    @test cb1.start_itr  == 0
    @test cb1.start_time == 0

    # test callback
    x1 = randn(1)
    x2 = randn(1)
    state1 = DummyState(0, 1, π, Dict("time"=>0, "Current step size"=>0.1, "x"=>ReSolver.OptVector(x1, 23.5)))
    state2 = DummyState(1, 0.8, π - 2, Dict("time"=>0.2, "Current step size"=>0.05, "x"=>ReSolver.OptVector(x2, 23.25)))
    @test !cb1(state1)
    @test        s.x == x1
    @test   s.period == 23.5
    @test      s.itr == 0
    @test     s.time == 0.0
    @test     s.step == 0.1
    @test s.residual == 1.0
    @test   s.g_norm == Float64(π)
    @test   s.other === nothing
    @test !cb1(state2)
    @test        s.x == x2
    @test   s.period == 23.25
    @test      s.itr == 1
    @test     s.time == 0.2
    @test     s.step == 0.05
    @test s.residual == 0.8
    @test   s.g_norm == Float64(π - 2)
    @test   s.other === nothing

    # construct new callback cache from previous trace
    cb2 = ReSolver.CallbackCache(s, OptOptions(verbose=false))
    @test cb2 isa ReSolver.CallbackCache{<:OptState{Vector{Float64}, Float64, Nothing}, <:OptOptions}
    @test cb2.start_itr == 1
    @test cb2.start_time == 0.2
    x1 = randn(1)
    x2 = randn(1)
    state3 = DummyState(0, 0.8, π - 2, Dict("time"=>0, "Current step size"=>0.05, "x"=>ReSolver.OptVector(x1, 23.25)))
    state4 = DummyState(1, 0.4, π - 3.1, Dict("time"=>0.5, "Current step size"=>0.03, "x"=>ReSolver.OptVector(x2, 23.12)))
    @test !cb2(state3)
    @test        s.x == x1
    @test   s.period == 23.25
    @test      s.itr == 1
    @test     s.time == 0.2
    @test     s.step == 0.05
    @test s.residual == 0.8
    @test   s.g_norm == Float64(π - 2)
    @test   s.other === nothing
    @test !cb2(state4)
    @test        s.x == x2
    @test   s.period == 23.12
    @test      s.itr == 2
    @test     s.time == 0.7
    @test     s.step == 0.03
    @test s.residual == 0.4
    @test   s.g_norm == Float64(π - 3.1)
    @test   s.other === nothing
end
