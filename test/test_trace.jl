@testset "Optimisation trace        " begin
    # construct traces
    t1 = OptTrace([0.0])
    t2 = OptTrace([0.0], x->x[1]^2)
    @test t1 isa OptTrace{Nothing}
    @test t2 isa OptTrace{Float64}

    # test interface
    @test length(t1) == length(t2) == 0
    push!(t1, [π], 0, 0, 0.5, 1, 1, 23.5, 0, 0)
    push!(t2, [π], 0, 0, 0.5, 1, 1, 23.5, 5, 15)
    @test length(t1) == length(t2) == 1

    @test itrs(t1)                       == [0]
    @test itrs(t2)                       == [5]
    @test times(t1)                      == [0]
    @test times(t2)                      == [15]
    @test steps(t1)     == steps(t2)     == [0.5]
    @test residuals(t1) == residuals(t2) == [1]
    @test g_norms(t1)   == g_norms(t2)   == [1]
    @test periods(t1)   == periods(t2)   == [23.5]
    @test others(t1)                     == []
    @test others(t2)                     == [π^2]

    @test t1[1] == (0, 0, 0.5, 1, 1, 23.5, nothing)
    @test t2[1] == (5, 15, 0.5, 1, 1, 23.5, π^2)
    push!(t1, [2π], 1, 0.1, 0.5, 1, 1, 23.25, 0, 0)
    push!(t2, [2π], 1, 0.1, 0.5, 1, 1, 23.25, 5, 15)
    @test length(t1) == length(t2) == 2
    @test t1[2] == (1, 0.1, 0.5, 1, 1, 23.25, nothing)
    @test t2[2] == (6, 15.1, 0.5, 1, 1, 23.25, 4*π^2)

    # check printing is correct
    # ReSolver.print_header(stdout, t1)
    # ReSolver.print_state(stdout, t1[1])
    # ReSolver.print_state(stdout, t1[2])
    # ReSolver.print_state(stdout, t2[1])
    # ReSolver.print_state(stdout, t2[2])
end
