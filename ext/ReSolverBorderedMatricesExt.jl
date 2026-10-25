# Banded block elimination of the coarse matrix of TwoGrid, with BorderedMatrices.jl. The interior
# blocks form a banded matrix of bandwidths (h + 1) r - 1; the last h blocks, which carry the cyclic
# wrap-around, and the parameters form a dense border (blockcyclic). The factorisation and its
# memory are linear in the number of blocks.
module ReSolverBorderedMatricesExt

using ReSolver, BorderedMatrices, LinearAlgebra

struct BandedCoarse{F}
    K::F   # factorisation of the bordered matrix
    n::Int # unknowns in its banded part
end

function ReSolver._factorise_coarse(::Val{:banded}, block, r::Int, Nb::Int, hb::Int,
                                    b::Matrix{Float64}, c::Matrix{Float64})
    q = size(b, 2)
    M = blockcyclic(block, r, Nb, hb, b, c, zeros(q, q))
    return BandedCoarse(lu(M), size(M.A, 1))
end

ReSolver.coarse_solve(C::BandedCoarse, y::Vector{Float64}) =
    collect(C.K \ BorderedVector(y[1:C.n], y[C.n + 1:end]))

ReSolver._coarse_bytes(C::BandedCoarse) = Base.summarysize(C.K)

end
