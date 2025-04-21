@testset "Optimisation vector       " begin
    # test construction
    x = ReSolver.OptVector(randn(1), rand())
    @test x isa ReSolver.OptVector{Vector{Float64}, 2}

    # test array interface
    @test size(x) == (2,)
    @test length(x) == 2
    @test eltype(x) == Float64
    @test x[1] == x.x[1]; @test x[2] == x.T
    @test similar(x) isa ReSolver.OptVector{Vector{Float64}, 2}
    @test copy(x) == ReSolver.OptVector(x.x, x.T)

    # broadcasting
    fun!(x, y, z) = @allocated z .= 2.0.*y .- x./5.0
    y = copy(x); z = similar(x)
    @test fun!(x, y, z) == 0
    @test z == ReSolver.OptVector(2.0.*y.x .- x.x./5.0, 2*y.T - x.T/5)
    @test z == 2.0.*y .- x./5.0
end
