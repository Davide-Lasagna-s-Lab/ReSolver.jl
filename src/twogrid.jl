# Two-grid preconditioner for the Newton system of the hookstep.
#
# The Newton system 𝒥 δx = b is solved by GMRES, right preconditioned. A preconditioner B given by
# the user, such as a diagonal Fourier preconditioner, is accurate where the linear part of the
# operator dominates, at small scales and high temporal frequencies, and inaccurate at large scales
# and low frequencies, where the nonlinear terms couple the modes: the slow directions that make
# GMRES stall. A two-grid method splits the work accordingly (Hackbusch 1985, Briggs et al. 2000,
# Trottenberg et al. 2001):
#
#     smoothing                 z ← z + θ B⁻¹ (b - 𝒥 z)     B as smoother: a preconditioned
#                                                           Richardson iteration
#     coarse-grid correction    z ← z + P K⁻¹ R (b - 𝒥 z)   the error left by the smoother, solved
#                                                           in a coarse space
#
# Coarse space. In space, the first r coordinates of each time slice, ordered from large to small
# scales: spectral truncation (Zang, Wong & Hussaini 1982). The coordinates are given by the user, as
# an isometry between the fields and real arrays of size nx × Ns, one column per time slice:
# `coords(c, a)` writes the coordinates of the field `a` into the matrix `c`, `coords(a)` returns
# them as a new matrix, and `field!(a, c)` writes the field of the coordinates `c` into `a`. In
# time, three choices:
#
#     time = :splines, tc = 1    every slice
#     time = :splines, tc > 1    periodic cubic B-splines on a grid tc times coarser, from the
#                                package PeriodicBSplines.jl, documented on its own
#                                (github.com/Davide-Lasagna-s-Lab/PeriodicBSplines.jl)
#     time = :fourier            the values at Nc ≈ Ns/tc equispaced coarse times (Nc odd), with
#                                spectral transfers: the prolongation is trigonometric interpolation
#                                (zero padding of the temporal Fourier series), the restriction its
#                                adjoint, the truncation of the series to |n| ≤ (Nc - 1)/2 sampled at
#                                the coarse times: an exact low-pass
#
# The prolongation P maps the coarse unknowns and the parameters to an orbit; the restriction is its
# adjoint, R = Pᵀ.
#
# Coarse operator. K = R 𝒥̃ P, the Galerkin projection of the Jacobian, with the time derivative
#
#     every slice    rediscretised by centred differences of order 2h between the slices,
#     B-splines      exact on the B-splines, whose derivative is known,
#     Fourier        rediscretised in the time domain: the spatial blocks evaluated at the coarse
#                    times (collocation; their trigonometric interpolation from the slices, exact
#                    for the band-limited orbit) and the time derivative by the Galerkin derivative
#                    of the B-splines used as a compact scheme on the coarse values,
#                    Σⱼ mⱼ y′_{i+j} = Σⱼ dⱼ y_{i+j}/Δ (PeriodicBSplines.mass_stencil and
#                    derivative_stencil), nearly spectral; the coarse equations are multiplied by
#                    the stencil m, which keeps K banded, and so is the restricted residual.
#                    The Galerkin K of the band-limited space would be dense in time, because the
#                    advection multiplies by the orbit, which couples all the temporal harmonics.
#
# bordered by the columns of the parameters (ω ∂s a, -∂ᵢ a) and the rows of the phase conditions
# (∂s a, ∂ᵢ a), restricted. The spatial operator, local in time, enters through its blocks at each
# slice, A_l = -R_l (L_l + Σᵢ cᵢ ∂ᵢ) P_l, obtained by probing: one application of the linearised
# operator to the field whose coordinates are e_α at every slice gives the column α of all the
# blocks at once, so building K costs r applications whatever the period. K is cyclic
# block-banded, with r × r blocks and h block diagonals on each side (3 for B-splines and Fourier),
# and is factorised directly: by sparse LU (`solver=:sparse`), or by the banded block elimination
# of BorderedMatrices.jl (`solver=:banded`, an extension), linear in time and memory in the number
# of blocks.
#
# Cycles. With `cycle=:multiplicative`, `pre` smoothing steps, the coarse-grid correction and
# `post` smoothing steps, the first from z = 0; each step after the first costs one Jacobian
# action. The default, pre = post = 2, is a symmetric two-grid cycle: smoothing on both sides of the
# coarse-grid correction is what makes the cycle robust (one-sided cycles stall). With `cycle=:additive`
# (tc = 1 only), z = B⁻¹ (I - P R) b + P K⁻¹ R b, no Jacobian action.
#
# The preconditioner applies the Jacobian, so it needs the System: the search calls the methods
# precondition!(out, B, F, p) and relinearise!(B, F, x) of precondition.jl, which pass it. It has
# no adjoint and serves the hookstep only.

using SparseArrays
using FFTW: rfft, irfft, fft, ifft
using PeriodicBSplines: collocation, mass_stencil, derivative_stencil

"""
    TwoGrid(B, coords, field!, r; tc=1, time=:splines, h=4, neutral=true, cycle=:multiplicative,
            pre=2, post=2, θ=1, solver=:sparse)

Two-grid preconditioner for the hookstep, with the preconditioner `B` as smoother. The coarse
space holds the first `r` spatial coordinates of each time slice. `coords(c, a)` writes the
coordinates of the field `a` into the `nx × Ns` matrix `c`, one column per time slice, ordered from
large to small scales and orthonormal in the inner product of the fields; `coords(a)` returns them
as a new matrix, and `field!(a, c)` is the inverse. With `neutral=true` the coarse space also
holds, at each slice, the profiles of the neutral directions, ∂s a and the drift derivatives.

In time, with `time=:splines`, the coarse space holds every slice (`tc=1`) or periodic cubic
B-splines on a time grid `tc` times coarser (`tc > 1`), and the coarse operator is the Galerkin
projection of the Jacobian, with the time derivative replaced by centred differences of order `2h`
when `tc=1`. With `time=:fourier`, it holds the values at `Nc ≈ Ns/tc` equispaced coarse times
(`Nc` odd), with spectral transfers in time (trigonometric interpolation, and its adjoint, the
truncation of the temporal Fourier series); the coarse operator is collocated at the coarse times,
with the Galerkin derivative of the B-splines as a compact scheme in time. Either way it is
cyclic block-banded, and factorised by sparse LU (`solver=:sparse`) or, after
`using BorderedMatrices`, by banded block elimination (`solver=:banded`).

`cycle=:multiplicative` applies `pre` smoothing steps with weight `θ`, the coarse-grid correction and
`post` smoothing steps; `cycle=:additive` adds the smoother, on the part of the residual outside
the coarse space, to the coarse correction, and needs every slice (`tc=1`, `time=:splines`).
Rebuilt about the orbit at every Newton iteration, by [`relinearise!`](@ref).

```julia
coords, field! = odd_sine_coordinates(g)                 # Kuramoto–Sivashinsky example
B = TwoGrid(KSPreconditioner(x; kind=:jacobian), coords, field!, 16; tc=2, solver=:banded)  # this
B = TwoGrid(KSPreconditioner(x; kind=:jacobian), coords, field!, 16; tc=2, time=:fourier)  # or this
F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B, project=odd!)
```
"""
mutable struct TwoGrid{S, CO, FI}
    smoother::S                  # B, a preconditioner of the search
      coords::CO                 # coords(c, a): coordinates of a field, nx × Ns
      field!::FI                 # field!(a, c): field of coordinates
           r::Int                # coarse coordinates per time slice
          tc::Int                # coarsening factor in time
        time::Symbol             # :splines (every slice if tc = 1) or :fourier
           h::Int                # half-width of the centred differences, tc = 1
       cycle::Symbol             # :multiplicative or :additive
         pre::Int                # smoothing steps before the coarse-grid correction
        post::Int                # smoothing steps after
           θ::Float64            # weight of the smoother
      solver::Symbol             # :sparse or :banded
           x::Any                # orbit about which K is built
           K::Any                # factorisation of the coarse matrix
          Pt::Any                # B-splines at the slices, Ns × Nc, or nothing for tc = 1
         PtT::Any                # its transpose, stored for the prolongation
     neutral::Bool               # add the neutral profiles ∂s a, ∂ᵢ a to the coarse basis
           V::Any                # the neutral profiles, nx × ne × Ns, or nothing
           c::Matrix{Float64}    # scratch coordinates
        work::Any                # two scratch orbits
      builds::Int                # applications of the linearised operator to build K, cumulative
       nodes::Int                # coarse unknowns
          Nc::Int                # coarse times, time = :fourier

    function TwoGrid(smoother, coords, field!, r::Int; tc::Int=1, time::Symbol=:splines, h::Int=4,
                     neutral::Bool=true, cycle::Symbol=:multiplicative, pre::Int=2, post::Int=2,
                     θ::Real=1.0, solver::Symbol=:sparse)
        cycle in (:multiplicative, :additive) ||
            throw(ArgumentError("cycle must be :multiplicative or :additive"))
        time in (:splines, :fourier) ||
            throw(ArgumentError("time must be :splines or :fourier"))
        cycle === :additive && (tc > 1 || time === :fourier) &&
            throw(ArgumentError("the additive cycle needs every slice: tc = 1, time = :splines"))
        solver in (:sparse, :banded) ||
            throw(ArgumentError("solver must be :sparse or :banded"))
        tc >= 1 || throw(ArgumentError("tc must be at least 1"))
        return new{typeof(smoother), typeof(coords), typeof(field!)}(
            smoother, coords, field!, r, tc, time, h, cycle, pre, post, θ, solver,
            nothing, nothing, nothing, nothing, neutral, nothing, zeros(0, 0), nothing, 0, 0, 0)
    end
end

# centred differences of order 2h for the first derivative: weights wⱼ of (u_{l+j} - u_{l-j})/Δs
function centred_weights(h::Int)
    A = [Float64(j)^(2i - 1) for i in 1:h, j in 1:h]
    return A \ [i == 1 ? 0.5 : 0.0 for i in 1:h]
end

# ---- periodic cubic B-splines in time: PeriodicBSplines.collocation(Ns, Nc) gives the values and
#      s-derivatives of the Nc splines centred at 2π m/Nc, at the Ns slices 2π l/Ns ----

# ---- spectral transfers in time, time = :fourier: Ns and Nc odd, so no Nyquist harmonic ----
# trigonometric interpolation of the rows of Y from Nc to Ns equispaced times
function _tprolong(Y::AbstractMatrix{Float64}, Ns::Int)
    Ŷ = rfft(Y, 2)
    Z = zeros(ComplexF64, size(Y, 1), Ns ÷ 2 + 1)
    Z[:, axes(Ŷ, 2)] .= Ŷ
    return irfft(Z, Ns, 2) .* (Ns / size(Y, 2))
end

# its adjoint: the temporal Fourier series truncated to |n| ≤ (Nc - 1)/2, at the Nc coarse times
_trestrict(Y::AbstractMatrix{Float64}, Nc::Int) = irfft(rfft(Y, 2)[:, 1:Nc ÷ 2 + 1], Nc, 2)

# the values at the Nc coarse times of the trigonometric interpolant of X[:, :, l], l = 1, …, Ns:
# every harmonic n of the slices contributes to the coarse harmonic n mod Nc
function _tsample(X::Array{Float64, 3}, Nc::Int)
    Ns = size(X, 3)
    X̂ = fft(X, 3)
    F = zeros(ComplexF64, size(X, 1), size(X, 2), Nc)
    for i in 1:Ns
        n = i <= Ns ÷ 2 + 1 ? i - 1 : i - 1 - Ns
        view(F, :, :, mod(n, Nc) + 1) .+= view(X̂, :, :, i)
    end
    return real.(ifft(F, 3)) .* (Nc / Ns)
end

# the coarse equations multiplied by the mass stencil of the compact derivative, time = :fourier
function _premultiply(Y::AbstractMatrix{Float64})
    m, Nb = mass_stencil(), size(Y, 2)
    Z = zeros(size(Y))
    for i in 1:Nb, j in -3:3
        view(Z, :, i) .+= m[j + 4] .* view(Y, :, mod(i - 1 + j, Nb) + 1)
    end
    return Z
end

nblocks(tg::TwoGrid) = tg.time === :fourier ? tg.Nc : tg.Pt === nothing ? size(tg.c, 2) : size(tg.Pt, 2)
ncoarse(tg::TwoGrid) = tg.r + (tg.V === nothing ? 0 : size(tg.V, 2))

# ---- restriction and prolongation, slice by slice ----
# the coarse coordinates of each slice: the first r coordinates, then the components along the
# neutral profiles V[:, e, l], which have no component on the first r
function _restrict_slices(tg::TwoGrid, c::Matrix{Float64})
    r, Ns = tg.r, size(c, 2)
    Y = zeros(ncoarse(tg), Ns)
    Y[1:r, :] .= view(c, 1:r, :)
    if tg.V !== nothing
        for l in 1:Ns, e in axes(tg.V, 2)
            Y[r + e, l] = dot(view(tg.V, :, e, l), view(c, :, l))
        end
    end
    return Y
end

function _prolong_slices!(c::Matrix{Float64}, tg::TwoGrid, Y::AbstractMatrix)
    r, Ns = tg.r, size(c, 2)
    fill!(c, 0)
    c[1:r, :] .= view(Y, 1:r, :)
    if tg.V !== nothing
        for l in 1:Ns, e in axes(tg.V, 2)
            view(c, :, l) .+= Y[r + e, l] .* view(tg.V, :, e, l)
        end
    end
    return c
end

# y = R p = Pᵀ p: the coarse coordinates of every slice, in time Pₜᵀ, then the parameters
function restrict(tg::TwoGrid, p::Orbit)
    tg.coords(tg.c, p.a)
    Y = _restrict_slices(tg, tg.c)
    tg.time === :fourier && (Y = _trestrict(Y, tg.Nc))
    tg.Pt === nothing || (Y = Y * tg.Pt)
    return vcat(vec(Y), p.p)
end

# out = P y
function prolong!(out::Orbit, tg::TwoGrid, y::AbstractVector)
    rc, Nb = ncoarse(tg), nblocks(tg)
    Y = reshape(view(y, 1:rc * Nb), rc, Nb)
    _prolong_slices!(tg.c, tg, tg.time === :fourier ? _tprolong(Y, size(tg.c, 2)) :
                               tg.Pt === nothing ? Y : Y * tg.PtT)
    tg.field!(out.a, tg.c)
    out.p .= view(y, rc * Nb + 1:length(y))
    return out
end

# ---- coarse matrix ----
function relinearise!(tg::TwoGrid, F, x::Orbit)
    relinearise!(tg.smoother, F, x)
    tg.x    = copy(x)
    tg.work = (similar(x), similar(x))
    a       = x.a
    tg.c    = tg.coords(a)
    nx, Ns  = size(tg.c)
    r, q    = tg.r, length(x.p)
    ω       = frequency(x)
    1 <= r < nx || throw(ArgumentError("r must be between 1 and $(nx - 1)"))
    Φ, LΦ, tmp = similar(a), similar(a), similar(a)

    # ---- neutral profiles: ∂s a and ∂ᵢ a at each slice, without their first r coordinates,
    #      orthonormalised slice by slice ----
    tg.V = nothing
    if tg.neutral
        ds = [(F.dds!, nothing); [(dd!, i) for (i, dd!) in enumerate(F.ddi)]]
        V  = zeros(nx, length(ds), Ns)
        for (e, (d!, _)) in enumerate(ds)
            d!(tmp, a)
            tg.coords(tg.c, tmp)
            V[r + 1:end, e, :] .= view(tg.c, r + 1:nx, :)
        end
        for l in 1:Ns                                  # Gram–Schmidt, diag(R) > 0: smooth in s
            Q, R_ = qr(V[:, :, l])
            Q = Matrix(Q)
            for e in axes(V, 2)
                if abs(R_[e, e]) > 1e-12 * max(1, norm(view(V, :, :, l)))
                    V[:, e, l] .= sign(R_[e, e]) .* view(Q, :, e)
                else                                   # no profile at this slice: any unit vector
                    V[:, e, l] .= 0; V[r + e, e, l] = 1
                end
            end
        end
        tg.V = V
    end
    rc = ncoarse(tg)

    # ---- spatial blocks at the slices, A_l = -R_l (L + Σᵢ cᵢ ∂ᵢ) P_l, rc × rc, by probing: the
    #      field with the coarse coordinates e_α at every slice, all slices at once ----
    A = zeros(rc, rc, Ns)
    for α in 1:rc
        Y = zeros(rc, Ns); Y[α, :] .= 1
        tg.field!(Φ, _prolong_slices!(tg.c, tg, Y))
        F.lin(LΦ, Φ)
        for (i, dd!) in enumerate(F.ddi)
            dd!(tmp, Φ)
            LΦ .+= drift(x, i) .* tmp
        end
        tg.coords(tg.c, LΦ)
        A[:, α, :] .= .-_restrict_slices(tg, tg.c)
    end
    tg.builds += rc

    # ---- border at the slices: columns of log ω and of the drift speeds, rows of the phase
    #      conditions, rc × Ns each ----
    bcol = Matrix{Float64}[]
    brow = Matrix{Float64}[]
    F.dds!(tmp, a)
    tg.coords(tg.c, tmp)
    d = _restrict_slices(tg, tg.c)
    push!(bcol, ω .* d); push!(brow, d)
    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, a)
        tg.coords(tg.c, tmp)
        d = _restrict_slices(tg, tg.c)
        push!(bcol, .-d); push!(brow, d)
    end

    # Gram matrix of the coarse bases of the slices l and k, Q_lᵀ Q_k: I on the first r
    function gram(l, k)
        G = Matrix{Float64}(I, rc, rc)
        if tg.V !== nothing
            G[r + 1:end, r + 1:end] .= transpose(view(tg.V, :, :, l)) * view(tg.V, :, :, k)
        end
        return G
    end

    if tg.time === :splines && tg.tc == 1
        # ---- every slice; the time derivative by centred differences, ω wⱼ/Δs Q_lᵀ Q_k ----
        tg.Pt, tg.PtT = nothing, nothing
        Nb, hb = Ns, tg.h
        Ns > 2hb || throw(ArgumentError("need more than 2h = $(2hb) time slices"))
        w  = ω .* centred_weights(hb) ./ (2π / Ns)
        block = function (l, k)
            l == k && return A[:, :, l]
            j = mod(k - l, Ns)
            return (j <= hb ? w[j] : -w[Ns - j]) .* gram(l, k)
        end
        b = reduce(hcat, [vec(B) for B in bcol])
        c = reduce(hcat, [vec(B) for B in brow])
    elseif tg.time === :splines
        # ---- B-splines φ_m in time, the field Q(s) Σ_m φ_m(s) y_m: its derivative is
        #      Q φ'_m y_m + Q' φ_m y_m, with Q_lᵀ Q_l = I and Q_lᵀ Q'_l nonzero only on the
        #      neutral profiles, Vᵀ V' ----
        Nb = round(Int, Ns / tg.tc)
        hb = 3
        Nb > 2hb || throw(ArgumentError("need more than $(2hb) coarse time points"))
        Pt, Pd = collocation(Ns, Nb)
        tg.Pt  = Pt
        QdQ = _neutral_derivative(tg, F, Φ, tmp)
        PtT, PdT = sparse(transpose(Pt)), sparse(transpose(Pd))
        tg.PtT = PtT
        Kb = _spline_blocks(A, QdQ, PtT, PdT, ω, hb, Nb)
        block = (l, k) -> Kb[:, :, mod(k - l + hb, Nb) + 1, l]
        b = reduce(hcat, [vec(B * Pt) for B in bcol])
        c = reduce(hcat, [vec(B * Pt) for B in brow])
    else
        # ---- Fourier: the values y_m at Nc odd coarse times s_m, the field Q(s) ỹ(s) with ỹ the
        #      trigonometric interpolant. R P = (Ns/Nc) I, and R 𝒥 P ≈ (Ns/Nc) (ω ∂s + A + ω QᵀQ′)
        #      collocated at the s_m, with ∂s the compact derivative M⁻¹D/Δ of the B-splines; the
        #      equations multiplied by M: block (m, m + j) = (Ns/Nc) (ω dⱼ/Δ I + mⱼ C(s_{m+j})),
        #      C = A + ω QᵀQ′ evaluated at the coarse times exactly from its values at the slices ----
        Nb = max(7, 2 * round(Int, (Ns / tg.tc - 1) / 2) + 1)
        Nb <= Ns || throw(ArgumentError("need at least $Nb time slices"))
        hb = 3
        tg.Pt, tg.PtT, tg.Nc = nothing, nothing, Nb
        Cc = _tsample(A .+ ω .* _neutral_derivative(tg, F, Φ, tmp), Nb)
        m, d = mass_stencil(), derivative_stencil()
        f, Δ = Ns / Nb, 2π / Nb
        block = function (l, k)
            j = mod(k - l + hb, Nb) - hb
            return f .* (ω * d[j + 4] / Δ .* Matrix{Float64}(I, rc, rc) .+ m[j + 4] .* view(Cc, :, :, k))
        end
        b = reduce(hcat, [vec(_premultiply(_trestrict(B, Nb))) for B in bcol])
        c = reduce(hcat, [vec(_trestrict(B, Nb)) for B in brow])
    end

    tg.K     = _factorise_coarse(Val(tg.solver), block, rc, Nb, hb, b, c)
    tg.nodes = rc * Nb + q
    return tg
end

# Q_lᵀ ∂s Q_l at the slices, rc × rc × Ns: nonzero only on the neutral profiles, Vᵀ V′
function _neutral_derivative(tg::TwoGrid, F, Φ, tmp)
    rc, Ns = ncoarse(tg), size(tg.c, 2)
    QdQ = zeros(rc, rc, Ns)
    tg.V === nothing && return QdQ
    for e in axes(tg.V, 2)
        fill!(tg.c, 0); tg.c .= view(tg.V, :, e, :)
        tg.field!(Φ, tg.c)
        F.dds!(tmp, Φ)
        tg.coords(tg.c, tmp)
        for l in 1:Ns, e′ in axes(tg.V, 2)
            QdQ[tg.r + e′, tg.r + e, l] = dot(view(tg.V, :, e′, l), view(tg.c, :, l))
        end
    end
    return QdQ
end

# the blocks of K with B-splines in time, K_mm' = Σ_l φ_m(s_l) (φ_m'(s_l) (A_l + ω Q_lᵀQ'_l) +
# ω φ'_m'(s_l) I), block (m, m + j) in Kb[:, :, j + hb + 1, m]
function _spline_blocks(A::Array{Float64, 3}, QdQ::Array{Float64, 3}, PtT::SparseMatrixCSC,
                        PdT::SparseMatrixCSC, ω::Float64, hb::Int, Nb::Int)
    rc, _, Ns = size(A)
    Kb  = zeros(rc, rc, 2hb + 1, Nb)
    idx, φs, dφs = rowvals(PtT), nonzeros(PtT), nonzeros(PdT)
    for l in 1:Ns, i in nzrange(PtT, l), i′ in nzrange(PtT, l)
        m, m′ = idx[i], idx[i′]
        jj = mod(m′ - m + hb, Nb) + 1                  # offset + hb + 1
        c₁ = φs[i] * φs[i′]
        c₂ = ω * φs[i] * dφs[i′]
        @inbounds for β in 1:rc, α in 1:rc
            Kb[α, β, jj, m] += c₁ * (A[α, β, l] + ω * QdQ[α, β, l])
        end
        @inbounds for α in 1:rc
            Kb[α, α, jj, m] += c₂
        end
    end
    return Kb
end

# K as a sparse matrix, factorised by sparse LU
function _factorise_coarse(::Val{:sparse}, block, r::Int, Nb::Int, hb::Int, b::Matrix, c::Matrix)
    q, n = size(b, 2), r * Nb
    I_, J_, V_ = Int[], Int[], Float64[]
    for l in 1:Nb, j in -hb:hb
        k = mod(l - 1 + j, Nb) + 1
        B = block(l, k)
        for α in 1:r, β in 1:r
            iszero(B[β, α]) && continue
            push!(I_, (l - 1) * r + β); push!(J_, (k - 1) * r + α); push!(V_, B[β, α])
        end
    end
    for e in 1:q, i in 1:n
        push!(I_, i); push!(J_, n + e); push!(V_, b[i, e])
        push!(I_, n + e); push!(J_, i); push!(V_, c[i, e])
    end
    return lu(sparse(I_, J_, V_, n + q, n + q))
end

# the banded block elimination is defined by the extension for BorderedMatrices.jl
_factorise_coarse(::Val{:banded}, args...) =
    throw(ArgumentError("solver=:banded needs `using BorderedMatrices`"))

coarse_solve(K, y::Vector{Float64}) = K \ y

# the right-hand side of the coarse equations, R times the residual; with time = :fourier the
# equations are multiplied by the mass stencil, and so is its field part
function _coarse_rhs(tg::TwoGrid, y::Vector{Float64})
    tg.time === :fourier || return y
    n = ncoarse(tg) * nblocks(tg)
    return vcat(vec(_premultiply(reshape(y[1:n], ncoarse(tg), nblocks(tg)))), y[n + 1:end])
end

"""
    coarse_bytes(tg::TwoGrid) -> bytes

Memory of the factorisation of the coarse matrix.
"""
coarse_bytes(tg::TwoGrid) = _coarse_bytes(tg.K)
_coarse_bytes(K::SparseArrays.UMFPACK.UmfpackLU) = 16 * nnz(K)
_coarse_bytes(K) = Base.summarysize(K)

# ---- the cycle ----
# out = M⁻¹ p, M the two-grid preconditioner about tg.x
function precondition!(out::Orbit, tg::TwoGrid, F, p::Orbit)
    z, (t, res) = out, tg.work
    if tg.cycle === :additive
        y = restrict(tg, p)
        res .= p .- prolong!(t, tg, y)
        precondition!(z, tg.smoother, F, res)
        z .+= prolong!(t, tg, coarse_solve(tg.K, y))
        F.project(z.a)
        return z
    end

    fill!(z.a, 0); fill!(z.p, 0)
    zero_z = true
    for _ in 1:tg.pre
        _smooth!(z, tg, F, p, t, res, zero_z)
        zero_z = false
    end

    # ---- coarse-grid correction ----
    _residual!(res, tg, F, p, z, zero_z)
    z .+= prolong!(t, tg, coarse_solve(tg.K, _coarse_rhs(tg, restrict(tg, res))))
    F.project(z.a)

    for _ in 1:tg.post
        _smooth!(z, tg, F, p, t, res, false)
    end
    return z
end

# res = p - 𝒥 z, one Jacobian action, or p if z = 0
function _residual!(res::Orbit, tg::TwoGrid, F, p::Orbit, z::Orbit, zero_z::Bool)
    zero_z && return (res .= p)
    jacobian!(res, F, tg.x, z)
    res .= p .- res
    return res
end

# z ← z + θ B⁻¹ (p - 𝒥 z)
function _smooth!(z::Orbit, tg::TwoGrid, F, p::Orbit, t::Orbit, res::Orbit, zero_z::Bool)
    _residual!(res, tg, F, p, z, zero_z)
    precondition!(t, tg.smoother, F, res)
    F.project(t.a)
    z .+= tg.θ .* t
    return z
end

precondition!(out::Orbit, tg::TwoGrid, p::Orbit) =
    throw(ArgumentError("TwoGrid applies the Jacobian: use it as the preconditioner of a System"))
precondition_adjoint!(out::Orbit, tg::TwoGrid, p::Orbit) =
    throw(ArgumentError("TwoGrid has no adjoint: it serves the hookstep only"))
