using PeriodicBSplines
using LinearAlgebra, SparseArrays, Test

@testset "PeriodicBSplines" begin
    @testset "cardinal spline" begin
        @test bspline(0.0) ≈ 2 / 3 && bspline(1.0) ≈ 1 / 6 && bspline(2.0) == 0 && bspline(-1.5) ≈ bspline(1.5)
        for u in -2.5:0.1:2.5                          # derivatives against centred differences
            δ = 1e-6
            @test dbspline(u) ≈ (bspline(u + δ) - bspline(u - δ)) / 2δ atol = 1e-8
            abs(abs(u) - 1) > 1e-9 && abs(u) > 1e-9 && @test d2bspline(u) ≈ (dbspline(u + δ) - dbspline(u - δ)) / 2δ atol = 1e-6
        end
        @test sum(bspline(u) for u in -2:0.001:2) * 0.001 ≈ 1 atol = 1e-6
    end
    @testset "collocation" begin
        Ns, Nc = 60, 12
        Φ, Φ′ = collocation(Ns, Nc)
        @test size(Φ) == (Ns, Nc) && all(sum(Φ .!= 0, dims=2) .<= 4)
        @test all(sum(Φ, dims=2) .≈ 1)                 # partition of unity
        @test maximum(abs, sum(Φ′, dims=2)) < 1e-12
        c = randn(Nc)
        @test Φ * c ≈ [evaluate(c, 2π * (l - 1) / Ns) for l in 1:Ns]
        @test_throws ArgumentError collocation(10, 3)
    end
    @testset "Gram matrices" begin
        Nc = 10; Ns = 4000
        Φ, Φ′ = collocation(Ns, Nc)
        M, D = gram(Nc), derivative_gram(Nc)
        @test Matrix(M) ≈ Matrix(transpose(Φ) * Φ) .* (2π / Ns) rtol = 1e-6
        @test Matrix(D) ≈ Matrix(transpose(Φ) * Φ′) .* (2π / Ns) atol = 1e-6
        @test D ≈ -transpose(D)
        @test sum(M) ≈ 2π                              # ∫ (Σ φ_m)² = 2π
        θ = 2π * 3 / Nc; e = cis.(θ .* (0:Nc - 1))
        @test M * e ≈ (2π / Nc) * mass_symbol(θ) .* e
        @test D * e ≈ derivative_symbol(θ) .* e
    end
    @testset "interpolation, fourth order" begin
        f(s) = exp(sin(s))
        err(Nc) = (c = interpolate(f.(2π .* (0:Nc - 1) ./ Nc));
                   maximum(abs(evaluate(c, s) - f(s)) for s in range(0, 2π, length=1001)))
        @test interpolate(f.(2π .* (0:15) ./ 16))[1] isa Float64
        @test 12 < err(32) / err(64) < 20
    end
    @testset "modified wavenumber" begin
        @test modified_wavenumber(0.1) ≈ 0.1 rtol = 1e-6
        @test abs(modified_wavenumber(π / 2) - π / 2) < 0.05
        @test abs(modified_wavenumber(Float64(π))) < 1e-12      # the shortest harmonic is not differentiated
    end
end
