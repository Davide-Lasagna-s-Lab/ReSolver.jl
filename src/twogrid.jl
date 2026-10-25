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
# time, either every slice (tc = 1), or periodic cubic B-splines on a grid tc times coarser
# (tc > 1). The prolongation P maps the coarse unknowns, the coefficients of the r modes at the Ns
# slices or of the B-splines, and the parameters, to an orbit; the restriction is its adjoint,
# R = Pᵀ, as for a Galerkin method.
#
# Coarse operator. K = R 𝒥̃ P, the Galerkin projection of the Jacobian, with the time derivative
#
#     tc = 1    rediscretised by centred differences of order 2h between the slices,
#     tc > 1    exact on the B-splines, whose derivative is known,
#
# bordered by the columns of the parameters (ω ∂s a, -∂ᵢ a) and the rows of the phase conditions
# (∂s a, ∂ᵢ a), restricted. The spatial operator, local in time, enters through its blocks at each
# slice, A_l = -R_l (L_l + Σᵢ cᵢ ∂ᵢ) P_l, obtained by probing: one application of the linearised
# operator to the field whose coordinates are e_α at every slice gives the column α of all the
# blocks at once, so building K costs r applications whatever the period. K is cyclic
# block-banded, with r × r blocks and h block diagonals on each side (3 for the cubic B-splines),
# and is factorised directly: by sparse LU (`solver=:sparse`), or by the banded block elimination
# of BorderedMatrices.jl (`solver=:banded`, an extension), linear in time and memory in the number
# of blocks.
#
# Cycles. With `cycle=:multiplicative`, `pre` smoothing steps, the coarse-grid correction and
# `post` smoothing steps, the first from z = 0; each step after the first costs one Jacobian
# action. The default, pre = post = 1, is the symmetric two-grid cycle. With `cycle=:additive`
# (tc = 1 only), z = B⁻¹ (I - P R) b + P K⁻¹ R b, no Jacobian action.
#
# The preconditioner applies the Jacobian, so it needs the System: the search calls the methods
# precondition!(out, B, F, p) and relinearise!(B, F, x) of precondition.jl, which pass it. It has
# no adjoint and serves the hookstep only.

using SparseArrays

"""
    TwoGrid(B, coords, field!, r; tc=1, h=4, cycle=:multiplicative, pre=1, post=1, θ=1,
            solver=:sparse)

Two-grid preconditioner for the hookstep, with the preconditioner `B` as smoother. The coarse
space holds the first `r` spatial coordinates of each time slice, at every slice (`tc=1`) or as
periodic cubic B-splines on a time grid `tc` times coarser (`tc > 1`). `coords(c, a)` writes the
coordinates of the field `a` into the `nx × Ns` matrix `c`, one column per time slice, ordered from
large to small scales and orthonormal in the inner product of the fields; `coords(a)` returns them
as a new matrix, and `field!(a, c)` is the inverse.

The coarse operator is the Galerkin projection of the Jacobian, with the time derivative replaced
by centred differences of order `2h` when `tc=1`. It is factorised by sparse LU (`solver=:sparse`)
or, after `using BorderedMatrices`, by banded block elimination (`solver=:banded`).
`cycle=:multiplicative` applies `pre` smoothing steps with weight `θ`, the coarse-grid correction and
`post` smoothing steps; `cycle=:additive` adds the smoother, on the part of the residual outside
the coarse space, to the coarse correction. Rebuilt about the orbit at every Newton iteration, by
[`relinearise!`](@ref).

```julia
coords, field! = odd_sine_coordinates(g)                 # Kuramoto–Sivashinsky example
B = TwoGrid(KSPreconditioner(x; kind=:jacobian), coords, field!, 12; tc=4)
F = System(KSNonlinear(g), KSLinearised(g), dds!, x; linearise!, B, project=odd!)
```
"""
mutable struct TwoGrid{S, CO, FI}
    smoother::S                  # B, a preconditioner of the search
      coords::CO                 # coords(c, a): coordinates of a field, nx × Ns
      field!::FI                 # field!(a, c): field of coordinates
           r::Int                # coarse coordinates per time slice
          tc::Int                # coarsening factor in time
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

    function TwoGrid(smoother, coords, field!, r::Int; tc::Int=1, h::Int=4, neutral::Bool=true,
                     cycle::Symbol=:multiplicative, pre::Int=1, post::Int=1, θ::Real=1.0,
                     solver::Symbol=:sparse)
        cycle in (:multiplicative, :additive) ||
            throw(ArgumentError("cycle must be :multiplicative or :additive"))
        cycle === :additive && tc > 1 &&
            throw(ArgumentError("the additive cycle needs tc = 1"))
        solver in (:sparse, :banded) ||
            throw(ArgumentError("solver must be :sparse or :banded"))
        tc >= 1 || throw(ArgumentError("tc must be at least 1"))
        return new{typeof(smoother), typeof(coords), typeof(field!)}(
            smoother, coords, field!, r, tc, h, cycle, pre, post, θ, solver,
            nothing, nothing, nothing, nothing, neutral, nothing, zeros(0, 0), nothing, 0, 0)
    end
end

# centred differences of order 2h for the first derivative: weights wⱼ of (u_{l+j} - u_{l-j})/Δs
function centred_weights(h::Int)
    A = [Float64(j)^(2i - 1) for i in 1:h, j in 1:h]
    return A \ [i == 1 ? 0.5 : 0.0 for i in 1:h]
end

# ---- periodic cubic B-splines in time ----
# the cardinal cubic B-spline, support [-2, 2], and its derivative
_bspline(u)  = (v = abs(u); v < 1 ? (4 - 6v^2 + 3v^3) / 6 : v < 2 ? (2 - v)^3 / 6 : 0.0)
_dbspline(u) = (v = abs(u); v < 1 ? -2u + 1.5u * v : v < 2 ? -sign(u) * (2 - v)^2 / 2 : 0.0)

# values and s-derivatives of the Nc B-splines centred at 2π m/Nc, at the Ns slices 2π l/Ns
function bsplines(Ns::Int, Nc::Int)
    Δ = 2π / Nc
    I_, J_, V_, D_ = Int[], Int[], Float64[], Float64[]
    for l in 1:Ns
        s  = 2π * (l - 1) / Ns
        m₀ = floor(Int, s / Δ)
        for m in m₀ - 1:m₀ + 2
            u = (s - Δ * m) / Δ
            push!(I_, l); push!(J_, mod(m, Nc) + 1)
            push!(V_, _bspline(u)); push!(D_, _dbspline(u) / Δ)
        end
    end
    return sparse(I_, J_, V_, Ns, Nc), sparse(I_, J_, D_, Ns, Nc)
end

nblocks(tg::TwoGrid) = tg.Pt === nothing ? size(tg.c, 2) : size(tg.Pt, 2)
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
    tg.Pt === nothing || (Y = Y * tg.Pt)
    return vcat(vec(Y), p.p)
end

# out = P y
function prolong!(out::Orbit, tg::TwoGrid, y::AbstractVector)
    rc, Nb = ncoarse(tg), nblocks(tg)
    Y = reshape(view(y, 1:rc * Nb), rc, Nb)
    _prolong_slices!(tg.c, tg, tg.Pt === nothing ? Y : Y * tg.PtT)
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
        for l in 1:Ns
            Q, R_ = qr(V[:, :, l])
            Q = Matrix(Q)
            for e in axes(V, 2)
                if abs(R_[e, e]) > 1e-12 * max(1, norm(view(V, :, :, l)))
                    V[:, e, l] .= view(Q, :, e)
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

    if tg.tc == 1
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
    else
        # ---- B-splines φ_m in time, the field Q(s) Σ_m φ_m(s) y_m: its derivative is
        #      Q φ'_m y_m + Q' φ_m y_m, with Q_lᵀ Q_l = I and Q_lᵀ Q'_l nonzero only on the
        #      neutral profiles, Vᵀ V' ----
        Nb = round(Int, Ns / tg.tc)
        hb = 3
        Nb > 2hb || throw(ArgumentError("need more than $(2hb) coarse time points"))
        Pt, Pd = bsplines(Ns, Nb)
        tg.Pt  = Pt
        QdQ = zeros(rc, rc, Ns)                        # Q_lᵀ ∂s Q_l
        if tg.V !== nothing
            for e in axes(tg.V, 2)
                fill!(tg.c, 0); tg.c .= view(tg.V, :, e, :)
                tg.field!(Φ, tg.c)
                F.dds!(tmp, Φ)
                tg.coords(tg.c, tmp)
                for l in 1:Ns, e′ in axes(tg.V, 2)
                    QdQ[r + e′, r + e, l] = dot(view(tg.V, :, e′, l), view(tg.c, :, l))
                end
            end
        end
        PtT, PdT = sparse(transpose(Pt)), sparse(transpose(Pd))
        tg.PtT = PtT
        Kb = _spline_blocks(A, QdQ, PtT, PdT, ω, hb, Nb)
        block = (l, k) -> Kb[:, :, mod(k - l + hb, Nb) + 1, l]
        b = reduce(hcat, [vec(B * Pt) for B in bcol])
        c = reduce(hcat, [vec(B * Pt) for B in brow])
    end

    tg.K     = _factorise_coarse(Val(tg.solver), block, rc, Nb, hb, b, c)
    tg.nodes = rc * Nb + q
    return tg
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
    z .+= prolong!(t, tg, coarse_solve(tg.K, restrict(tg, res)))
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
