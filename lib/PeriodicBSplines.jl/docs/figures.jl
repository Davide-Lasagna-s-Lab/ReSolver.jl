# Data of the figures of the documentation, computed with the package; plotted by figures.py.
#     julia --project=docs docs/figures.jl && python3 docs/figures.py
using PeriodicBSplines, DelimitedFiles, SparseArrays
const D = joinpath(@__DIR__, "src", "assets", "data")
mkpath(D)

# the cardinal spline and its derivatives
u = collect(range(-2.5, 2.5, length=501))
writedlm(joinpath(D, "cardinal.txt"), [u bspline.(u) dbspline.(u) d2bspline.(u)])

# the periodic splines, Nc = 8, on a fine grid, and their sum
Φ, Φ′ = collocation(400, 8)
writedlm(joinpath(D, "periodic.txt"), [2π .* (0:399) ./ 400 Matrix(Φ) sum(Φ, dims=2)])

# collocation and Gram matrices, Ns = 40, Nc = 8
Φ, Φ′ = collocation(40, 8)
writedlm(joinpath(D, "collocation.txt"), Matrix(Φ))
writedlm(joinpath(D, "gram.txt"), Matrix(gram(8)))
writedlm(joinpath(D, "derivative_gram.txt"), Matrix(derivative_gram(8)))

# interpolation error against the number of splines: a smooth function and harmonics
f(s) = exp(sin(s))
err(g, Nc) = (c = interpolate(g.(2π .* (0:Nc - 1) ./ Nc)); maximum(abs(evaluate(c, s) - g(s)) for s in range(0, 2π, length=2001)))
Ncs = [8, 12, 16, 24, 32, 48, 64, 96, 128]
writedlm(joinpath(D, "convergence.txt"), [Ncs [err(f, N) for N in Ncs]])
# harmonics cos(n s) with Nc/n points per period
ppp = collect(range(2.2, 16, length=60))
rows = Float64[]
for p in ppp
    n = 6; Nc = round(Int, n * p)
    push!(rows, err(s -> cos(n * s), Nc))
end
writedlm(joinpath(D, "harmonics.txt"), [round.(Int, 6 .* ppp) ./ 6 rows])

# modified wavenumber of the Galerkin derivative, against centred differences of order 2, 4, 8
θ = collect(range(0, π, length=200))
fd(h, t) = (A = [Float64(j)^(2i - 1) for i in 1:h, j in 1:h]; w = A \ [i == 1 ? 0.5 : 0.0 for i in 1:h]; 2 * sum(w[j] * sin(j * t) for j in 1:h))
writedlm(joinpath(D, "wavenumber.txt"), [θ modified_wavenumber.(θ) fd.(1, θ) fd.(2, θ) fd.(4, θ)])
println("data written to ", D)
