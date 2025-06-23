@testset "Optimisation interface    " begin
    # construct initial condition and system
    x0 = [1.0]; T0 = 1
    RdR! = Residual(x0,
                    (dxdt, x)->(dxdt[1] = 0.0; dxdt),
                    (F, x)->(F[1] = x[1]^2; F),
                    (G, x, y)->(G[1] = 2*x[1]*y[1]))
    t = OptTrace(x0)
    opts = OptOptions(verbose=false,
                      maxiter=1000,
                      alg=GradientDescent(linesearch=LineSearches.Static(),
                                          alphaguess=LineSearches.InitialStatic(alpha=0.499)))

    # perform a couple of optimisations
    x, T, dict = optimise!(copy(x0), T0, RdR!, t, opts=opts)
    x, T, dict = optimise!(x, T, RdR!, t, opts=opts)

    @test x[1] < 2e-3
    @test T == T0 # test system doesn't modify period in any way
    @test length(t) == 2002
    @test itrs(t) == [collect(0:1000); collect(1000:2000)]
end
