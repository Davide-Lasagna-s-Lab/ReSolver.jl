@testset "Optimisation trace        " begin
    # construct traces
    s1 = OptState([0.0])
    s2 = OptState([0.0], x->x[1]^2)
    @test s1 isa OptState{Vector{Float64}, Float64, Nothing}
    @test s2 isa OptState{Vector{Float64}, Float64, Float64}

    # test interface
    ReSolver.update!(s1, [Float64(π)], 23.5, 0, 0, 0.5, 1.0, 1.0, 0, 0)
    ReSolver.update!(s2, [Float64(π)], 23.5, 0, 0, 0.5, 1.0, 1.0, 5, 15)

    @test ReSolver.x(s1)    == ReSolver.x(s2)    == [Float64(π)]
    @test period(s1)        == period(s2)        == 23.5
    @test itr(s1)                                == 0
    @test itr(s2)                                == 5
    @test ReSolver.time(s1)                      == 0
    @test ReSolver.time(s2)                      == 15
    @test ReSolver.step(s1) == ReSolver.step(s2) == 0.5
    @test residual(s1)      == residual(s2)      == 1
    @test g_norm(s1)        == g_norm(s2)        == 1
    @test other(s1)                             === nothing
    @test other(s2)                              == π^2

    ReSolver.update!(s1, [2π], 23.25, 1, 0.1, 0.5, 1.0, 1.0, 0, 0)
    ReSolver.update!(s2, [2π], 23.25, 1, 0.1, 0.5, 1.0, 1.0, 5, 15)
    @test ReSolver.x(s1)    == ReSolver.x(s2)    == [Float64(2π)]
    @test period(s1)        == period(s2)        == 23.25
    @test itr(s1)                                == 1
    @test itr(s2)                                == 6
    @test ReSolver.time(s1)                      == 0.1
    @test ReSolver.time(s2)                      == 15.1
    @test ReSolver.step(s1) == ReSolver.step(s2) == 0.5
    @test residual(s1)      == residual(s2)      == 1
    @test g_norm(s1)        == g_norm(s2)        == 1
    @test other(s1)                             === nothing
    @test other(s2)                              == Float64(2π)^2

    # check printing is correct
    # ReSolver.print_header(stdout)
    # ReSolver.print_state(stdout, s1)
    # ReSolver.print_state(stdout, s2)
end
