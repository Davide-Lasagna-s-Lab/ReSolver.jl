@testset "Residual interface        " begin
    # test construction
    RdR! = Residual([0.0],
                    (dxdt, x)->(dxdt[1] = 0.0; dxdt),
                    (F, x)->(F[1] = x[1]^2; F),
                    (G, x, y)->(G[1] = -2*x[1]*y[1]))
    @test RdR! isa Residual{Vector{Float64}}

    # test correctness
    dRdx = randn(1)
    @test RdR!(dRdx, [0.0], rand()) == (0, [0.0], 0.0)
    @test RdR!(dRdx, [1.0], rand()) == (0.5, [2.0], 0.0)
    x = randn(1)
    @test RdR!(dRdx, x, rand()) == (abs2(x[1]^2)/2, [2*x[1]^3], 0.0)

    # test allocation
    fun1(x, T) = @allocated RdR!(x, T)
    fun2(dRdx, x, T) = @allocated RdR!(dRdx, x, T)
    @test fun1(x, rand()) == fun2(dRdx, x, rand()) == 0
end
