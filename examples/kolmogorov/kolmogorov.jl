# Two-dimensional Kolmogorov flow in space-time form, for ReSolver: the vorticity equation
#
#     ∂t ω + u ∂x ω + v ∂y ω = (1/Re) ∇²ω - kf cos(kf y),     (x, y) ∈ [0, 2π)² periodic,
#
# with velocity u = -∂y ψ, v = ∂x ψ from the streamfunction, ∇²ψ = ω, with the conventions of
# OpenKolmogorovFlow.jl (and of Chandler & Kerswell 2013).
#
# The state is a Fourier field, as an FTField of OpenKolmogorovFlow.jl: the coefficients ω̂(kx, ky)
# of the active modes 0 ≤ kx ≤ n, |ky| ≤ n, normalised so that ω(x, y) = Σ ω̂ exp(i(kx x + ky y)),
# with the mean mode kx = ky = 0 always zero. A space-time field, a KFField, holds the state at the
# Ns rescaled times s_l = 2π (l - 1)/Ns, Ns odd: an (n+1) × (2n+1) × Ns complex array, kx along the
# first dimension, ky along the second in FFT order (0, 1, …, n, -n, …, -1), s along the third.
# Derivatives in space are multiplications of the coefficients; the derivative in s is computed by
# FFTs along s. Products are computed on the physical grid of (2m+2)² points, m = n + n÷2, which
# dealiases them by the 3/2 rule, as in OpenKolmogorovFlow.jl, and truncated back to the active
# modes. The inner product is the mean over space and rescaled time of the products of the fields,
# Σ over the coefficients of the full spectrum, so that the derivatives are exactly skew-adjoint and
# the adjoint operator is exact.
#
# Relative periodic orbits. Translations along x are a continuous symmetry, handled as in the other
# examples by a drift speed. Translations along y by multiples of 2π/kf, the forcing wavelength, are
# a discrete symmetry: an orbit may repeat after a period up to such a shift ℓy,
# ω(x, y, t + T) = ω(x + ℓx, y + ℓy, t). The grid keyword `yshift` = ℓy makes the space-time field
# periodic up to that shift, â(kx, ky, s + 2π) = exp(i ky ℓy) â(kx, ky, s): the derivative in s acts
# on the periodic part exp(-i φ s) â, φ = ky ℓy/2π, as a multiplication by i(n + φ) on its Fourier
# modes n. Only ∂s and the preconditioner change; ∂s stays exactly skew-adjoint.
#
# The model runs on the CPU or on a GPU: the arrays of the grid and of the fields are built with the
# constructor `array` given to the grid, `Array` by default or e.g. `CuArray` with CUDA.jl loaded,
# the FFTs go through AbstractFFTs, and the operators use broadcasting and views only, no scalar
# indexing. The initial guess is built on the CPU and copied to the device.
#
# The file provides what a ReSolver System needs, and the pieces to build an initial guess:
#
#     KFGrid(n, Ns; Re, kf, yshift,    grid: wavenumbers, mask, FFT plans, on the device
#            array)
#     KFField(g)                       space-time Fourier field, zero
#     ddx!(out, ω), dds!(out, ω)       derivatives along x and in rescaled time s
#     KFNonlinear, KFLinearised        N(ω), its linearisation and adjoint
#     linearise!(op, ω)                linearisation point of KFLinearised
#     KFPreconditioner                 diagonal preconditioner in space-time Fourier
#     physical(ω, l)                   the vorticity at the rescaled time s_l on the physical grid
#     integrate, recurrence            time integration (ETDRK4) and near-recurrences

using FFTW
using LinearAlgebra
using ReSolver

import ReSolver: precondition!, precondition_adjoint!


# ============================================================================ #
# Grid                                                                         #
# ============================================================================ #

# active and storage cutoffs, and points of the physical grid, as in OpenKolmogorovFlow.jl
up_dealias_size(n::Int) = n + n >> 1

struct KFGrid{R, C, T, FP, IP, FS, IS, A}
          n::Int     # active cutoff
          m::Int     # storage cutoff, m = n + n÷2
          M::Int     # physical points per direction, 2m + 2
         Ns::Int     # points in rescaled time, odd
         Re::Float64 # Reynolds number
         kf::Int     # forcing wavenumber
         kx::R       # wavenumbers along x, 0, …, n, as an (n+1) × 1 × 1 array
         ky::R       # wavenumbers along y in FFT order, as a 1 × (2n+1) × 1 array
          s::R       # temporal wavenumbers in FFT order, as a 1 × 1 × Ns array
     yshift::Float64 # shift along y after one period, ℓy
          σ::R       # temporal wavenumbers of the periodic part, n + ky ℓy/2π, 1 × (2n+1) × Ns
      twist::T       # exp(i ky ℓy s/2π), the phases of the shift, 1 × (2n+1) × Ns
       mask::R       # 0 on the mean mode, 1 elsewhere
     invlap::R       # inverse Laplacian -1/(kx² + ky²), zero on the mean mode
    forcing::C       # coefficients of the forcing -kf cos(kf y)
    physfwd::FP      # real FFT over (x, y), for every s, physical grid → padded coefficients
    physinv::IP      # its unnormalised inverse
     timefwd::FS     # FFT along s of the active coefficients
     timeinv::IS     # its unnormalised inverse
      array::A       # constructor of the arrays on the device, Array or e.g. CuArray

    function KFGrid(n::Int, Ns::Int; Re::Real=40, kf::Int=4, yshift::Real=0, array=Array)

        # ---- input checks ----
        isodd(Ns) ||
            throw(ArgumentError("Ns must be odd"))
        kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be active, kf ≤ n"))

        # ---- sizes and wavenumbers ----
        m  = up_dealias_size(n)
        M  = 2m + 2
        kx = reshape(collect(0.0:n), :, 1, 1)
        ky = reshape(Float64[0:n; -n:-1], 1, :, 1)
        s  = reshape(fftfreq(Ns, Ns), 1, 1, :)

        # ---- shifted temporal wavenumbers and phases of the shift along y ----
        φ     = ky .* (yshift / 2π)
        σ     = s .+ φ
        twist = cis.(φ .* reshape((0:Ns - 1) .* (2π / Ns), 1, 1, :))

        # ---- mask of the mean mode and inverse Laplacian ----
        k²     = kx.^2 .+ ky.^2
        mask   = @. Float64(k² != 0)
        invlap = @. ifelse(k² == 0, 0.0, -1 / k²)

        # ---- forcing -kf cos(kf y): coefficients -kf/2 at (0, ±kf), at every s ----
        forcing = zeros(ComplexF64, n + 1, 2n + 1, Ns)
        forcing[1, kf + 1, :]      .= -kf / 2
        forcing[1, 2n + 2 - kf, :] .= -kf / 2

        # ---- FFT plans on the device ----
        u = array(zeros(M, M, Ns))
        P = array(zeros(ComplexF64, m + 2, M, Ns))
        a = array(zeros(ComplexF64, n + 1, 2n + 1, Ns))

        physfwd = plan_rfft(u, (1, 2))
        physinv = plan_brfft(P, M, (1, 2))
        timefwd = plan_fft(a, 3)
        timeinv = plan_bfft(a, 3)

        dev = array.((kx, ky, s))

        return new{typeof(dev[1]),
                   typeof(a),
                   typeof(array(twist)),
                   typeof(physfwd),
                   typeof(physinv),
                   typeof(timefwd),
                   typeof(timeinv),
                   typeof(array)}(n,
                                  m,
                                  M,
                                  Ns,
                                  Re,
                                  kf,
                                  dev...,
                                  Float64(yshift),
                                  array(σ),
                                  array(twist),
                                  array(mask),
                                  array(invlap),
                                  array(forcing),
                                  physfwd,
                                  physinv,
                                  timefwd,
                                  timeinv,
                                  array)
    end
end

# the active coefficients zero-padded to the coefficients of the physical grid, and back, with the
# normalisation of the coefficients; the ky ≥ 0 and ky < 0 blocks are copied as views
function _pad!(P::AbstractArray{ComplexF64, 3}, a::AbstractArray{ComplexF64, 3}, g::KFGrid)
    n, M = g.n, g.M

    fill!(P, 0)
    view(P, 1:n + 1, 1:n + 1, :)     .= view(a, :, 1:n + 1, :)
    view(P, 1:n + 1, M - n + 1:M, :) .= view(a, :, n + 2:2n + 1, :)

    return P
end

function _truncate!(a::AbstractArray{ComplexF64, 3}, P::AbstractArray{ComplexF64, 3}, g::KFGrid)
    n, M = g.n, g.M

    view(a, :, 1:n + 1, :)      .= view(P, 1:n + 1, 1:n + 1, :) ./ M^2
    view(a, :, n + 2:2n + 1, :) .= view(P, 1:n + 1, M - n + 1:M, :) ./ M^2

    return a
end


# ============================================================================ #
# Fields                                                                       #
# ============================================================================ #

"""
    KFField(g, data=zeros)

Space-time Fourier field on the grid `g`: the active coefficients of the vorticity at the Ns rescaled
times, an (n+1) × (2n+1) × Ns complex array on the device of the grid.
"""
struct KFField{G<:KFGrid, D<:AbstractArray{ComplexF64, 3}} <: AbstractArray{ComplexF64, 3}
    data::D # active coefficients at the rescaled times
       g::G # grid, shared

    KFField(g::G, data::D=g.array(zeros(ComplexF64, g.n + 1, 2g.n + 1, g.Ns))) where {G<:KFGrid, D} =
        new{G, D}(data, g)
end

# ---- array interface ----
Base.size(u::KFField)                    = size(u.data)
Base.getindex(u::KFField, I::Int...)     = u.data[I...]
Base.setindex!(u::KFField, v, I::Int...) = (u.data[I...] = v)
Base.IndexStyle(::Type{<:KFField})       = IndexLinear()

Base.similar(u::KFField) = KFField(u.g, similar(u.data))
Base.copy(u::KFField)    = KFField(u.g, copy(u.data))

# Mean over space and rescaled time of the products of the fields: by Parseval, the sum over the full
# spectrum, where each stored mode with kx > 0 stands for itself and its conjugate, kx < 0
function LinearAlgebra.dot(u::KFField, v::KFField)
    all  = real(dot(u.data, v.data))
    zero = real(dot(view(u.data, 1, :, :), view(v.data, 1, :, :)))

    return (2all - zero) / u.g.Ns
end

LinearAlgebra.norm(u::KFField) = sqrt(dot(u, u))

# ---- broadcasting keeps the field type, and runs on the arrays of the device ----
Base.BroadcastStyle(::Type{<:KFField}) = Broadcast.ArrayStyle{KFField}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{KFField}}, ::Type{ComplexF64}) =
    similar(_kf_field(bc))

# the broadcast with the fields replaced by their arrays, executed by the array type, e.g. as a GPU
# kernel
function Base.copyto!(dest::KFField, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{KFField}})
    copyto!(dest.data, Broadcast.instantiate(_kf_unwrap(bc)))
    return dest
end

_kf_unwrap(bc::Broadcast.Broadcasted) = Broadcast.Broadcasted(bc.f, map(_kf_unwrap, bc.args))
_kf_unwrap(u::KFField)                = u.data
_kf_unwrap(x)                         = x

_kf_field(bc::Broadcast.Broadcasted) = _kf_field(bc.args)
_kf_field(args::Tuple)               = _kf_field(_kf_field(args[1]), Base.tail(args))
_kf_field(u::KFField, rest)          = u
_kf_field(::Any, rest)               = _kf_field(rest)
_kf_field(u)                         = u

# the vorticity at the rescaled time s_l on the physical grid of (2m+2)² points, on the CPU
function physical(u::KFField, l::Int)
    g = u.g
    a = Array(u.data)[:, :, l]
    P = zeros(ComplexF64, g.m + 2, g.M)

    P[1:g.n + 1, 1:g.n + 1]           .= a[:, 1:g.n + 1]
    P[1:g.n + 1, g.M - g.n + 1:g.M]   .= a[:, g.n + 2:2g.n + 1]

    return brfft(P, g.M)
end


# ============================================================================ #
# Derivatives                                                                  #
# ============================================================================ #

# out = ∂x u, a multiplication of the coefficients
ddx!(out::KFField, u::KFField) = (out.data .= im .* u.g.kx .* u.data; out)

# out = ∂s u, by FFTs along s of the periodic part exp(-i φ s) u
function dds!(out::KFField, u::KFField)
    g = u.g

    out.data .= conj.(g.twist) .* u.data
    out.data .= g.timefwd * out.data
    out.data .*= im .* g.σ ./ g.Ns
    out.data .= g.twist .* (g.timeinv * out.data)

    return out
end


# ============================================================================ #
# Operators                                                                    #
# ============================================================================ #

# Work arrays of the operators: four fields on the physical grid, four arrays of padded coefficients
# and two of active coefficients, on the device of the grid.
function _work(g::KFGrid)
    phys   = ntuple(i -> g.array(zeros(g.M, g.M, g.Ns)), 4)
    padded = ntuple(i -> g.array(zeros(ComplexF64, g.m + 2, g.M, g.Ns)), 4)
    active = ntuple(i -> g.array(zeros(ComplexF64, g.n + 1, 2g.n + 1, g.Ns)), 2)

    return phys, padded, active
end

# Velocity and vorticity gradient of the coefficients a on the physical grid: u = -∂y ψ, v = ∂x ψ,
# ∂x ω, ∂y ω, with ψ̂ = invlap â, written into the four physical arrays `phys`.
function _velocity_gradient!(phys, padded, b, a, g::KFGrid)
    multipliers = (-im .* g.ky .* g.invlap, im .* g.kx .* g.invlap, im .* g.kx, im .* g.ky)

    for (u, P, μ) in zip(phys, padded, multipliers)
        b .= μ .* a
        _pad!(P, b, g)
        mul!(u, g.physinv, P)
    end

    return phys
end

# the coefficients of the product field p on the physical grid, truncated to the active modes
function _coefficients!(a, P, p, g::KFGrid)
    mul!(P, g.physfwd, p)
    return _truncate!(a, P, g)
end

# N(ω) = -(u ∂x ω + v ∂y ω) + (1/Re) ∇²ω - kf cos(kf y), on the active modes, mean excluded
struct KFNonlinear{G<:KFGrid, W}
       g::G # grid
    work::W # work arrays

    KFNonlinear(g::G) where {G<:KFGrid} = (w = _work(g); new{G, typeof(w)}(g, w))
end

function (op::KFNonlinear)(out::KFField, ω::KFField)
    g                    = op.g
    phys, padded, active = op.work
    u, v, ωx, ωy         = phys

    # ---- velocity and vorticity gradient on the physical grid ----
    _velocity_gradient!(phys, padded, active[1], ω.data, g)

    # ---- advection, dealiased ----
    u .= .-(u .* ωx .+ v .* ωy)
    _coefficients!(active[2], padded[1], u, g)

    # ---- viscous term and forcing ----
    out.data .= g.mask .* (active[2] .- (g.kx.^2 .+ g.ky.^2) .* ω.data ./ g.Re .+ g.forcing)

    return out
end

# Linearisation about Ω, and its adjoint in the inner product of the fields:
#
#     L v  = -(U ∂x v + V ∂y v) - (u_v ∂x Ω + v_v ∂y Ω) + (1/Re) ∇²v
#     L⁺ w =  ∂x(U w) + ∂y(V w) + ∇⁻²(∂x(w ∂y Ω) - ∂y(w ∂x Ω)) + (1/Re) ∇²w
#
# with (U, V) the velocity of Ω and (u_v, v_v) that of v, products on the physical grid and results
# truncated to the active modes. The advection is in advective form in L and in conservative form
# in L⁺, which makes L⁺ the exact adjoint of the discrete L.
struct KFLinearised{G<:KFGrid, B, W}
          g::G    # grid
    adjoint::Bool # false: L, true: L⁺
       base::B    # U, V, ∂x Ω, ∂y Ω on the physical grid, set by linearise!
       work::W    # work arrays

    function KFLinearised(g::G; adjoint::Bool=false) where {G<:KFGrid}
        base = ntuple(i -> g.array(zeros(g.M, g.M, g.Ns)), 4)
        work = _work(g)

        return new{G, typeof(base), typeof(work)}(g, adjoint, base, work)
    end
end

function linearise!(op::KFLinearised, Ω::KFField)
    _, padded, active = op.work
    _velocity_gradient!(op.base, padded, active[1], Ω.data, op.g)

    return op
end

function (op::KFLinearised)(out::KFField, v::KFField)
    g                    = op.g
    U, V, Ωx, Ωy         = op.base
    phys, padded, active = op.work
    a, b, c, d           = phys

    if op.adjoint
        # ---- the argument w on the physical grid ----
        _pad!(padded[1], v.data, g)
        mul!(d, g.physinv, padded[1])

        # ---- conservative advection by (U, V) and transposed velocity term ----
        a .= U .* d
        _coefficients!(active[1], padded[1], a, g)
        out.data .= im .* g.kx .* active[1]

        a .= V .* d
        _coefficients!(active[1], padded[1], a, g)
        out.data .+= im .* g.ky .* active[1]

        a .= d .* Ωy
        _coefficients!(active[1], padded[1], a, g)
        out.data .+= g.invlap .* (im .* g.kx .* active[1])

        a .= d .* Ωx
        _coefficients!(active[1], padded[1], a, g)
        out.data .-= g.invlap .* (im .* g.ky .* active[1])
    else
        # ---- advective form: by (U, V), and of Ω by the velocity of v ----
        _velocity_gradient!(phys, padded, active[1], v.data, g)

        a .= .-(U .* c .+ V .* d .+ a .* Ωx .+ b .* Ωy)
        _coefficients!(active[2], padded[1], a, g)
        out.data .= active[2]
    end

    # ---- viscous term, mean excluded ----
    out.data .= g.mask .* (out.data .- (g.kx.^2 .+ g.ky.^2) .* v.data ./ g.Re)

    return out
end


# ============================================================================ #
# Preconditioner                                                               #
# ============================================================================ #

# Diagonal preconditioner in space-time Fourier, frozen at ω₀ and c₀: the coefficient of the mode
# (kx, ky, n) is divided by one of two multipliers,
#
#     kind = :jacobian    i(ω₀ n - c₀ kx) + (kx² + ky²)/Re, the linear space-time operator about
#                         ω = 0
#     kind = :linear      1 + |i(ω₀ n - c₀ kx) + (kx² + ky²)/Re|, its size
#
# with 1 on the mean mode, which is excluded. The parameters are divided by ‖ω₀ ∂s ω₀‖ and
# ‖∂x ω₀‖, their Gauss–Newton curvatures.
struct KFPreconditioner{C}
    β::C               # multipliers, (n+1) × (2n+1) × Ns, on the device
    s::Vector{Float64} # parameter scales

    function KFPreconditioner(x₀::Orbit; kind::Symbol=:jacobian)

        # ---- input checks ----
        kind in (:jacobian, :linear) ||
            throw(ArgumentError("kind must be :jacobian or :linear"))

        a₀ = x₀.a
        g  = a₀.g
        ω₀ = ReSolver.frequency(x₀)
        c₀ = length(x₀.p) > 1 ? x₀.p[2] : 0.0

        # ---- field; 1 on the mean mode ----
        A = @. im * (ω₀ * g.σ - c₀ * g.kx) + (g.kx^2 + g.ky^2) / g.Re
        B = kind === :jacobian ? A : @.(1 + abs(A) + 0im)
        β = @. ifelse(g.mask == 0, one(B), B)

        # ---- parameters ----
        ∂a = similar(a₀)
        s  = [norm(ω₀ .* dds!(∂a, a₀))]
        length(x₀.p) > 1 && push!(s, norm(ddx!(∂a, a₀)))

        return new{typeof(β)}(β, s)
    end
end

# B is diagonal in space-time Fourier, on the periodic part exp(-i φ s) of the field: B⁻⁺ divides by
# the complex conjugate multipliers
function precondition!(out::Orbit, B::KFPreconditioner, p::Orbit)
    g = p.a.g

    out.a.data .= conj.(g.twist) .* p.a.data
    out.a.data .= g.timefwd * out.a.data
    out.a.data .*= g.mask ./ (B.β .* g.Ns)
    out.a.data .= g.twist .* (g.timeinv * out.a.data)

    out.p .= p.p ./ B.s

    return out
end

function precondition_adjoint!(out::Orbit, B::KFPreconditioner, p::Orbit)
    g = p.a.g

    out.a.data .= conj.(g.twist) .* p.a.data
    out.a.data .= g.timefwd * out.a.data
    out.a.data .*= g.mask ./ (conj.(B.β) .* g.Ns)
    out.a.data .= g.twist .* (g.timeinv * out.a.data)

    out.p .= p.p ./ B.s

    return out
end


# ============================================================================ #
# Time integration and near-recurrences, for initial guesses, on the CPU       #
# ============================================================================ #

# Integrate a state ω̂₀ (the (n+1) × (2n+1) active coefficients) for nsteps steps of size dt with
# ETDRK4 (Kassam & Trefethen 2005), products dealiased on the physical grid, returning the states
# every `every` steps as the slices of an (n+1) × (2n+1) × (nsteps ÷ every + 1) array.
function integrate(ω̂₀::AbstractMatrix, g::KFGrid, dt::Real, nsteps::Int; every::Int=1)
    n, m, M = g.n, g.m, g.M
    kx      = Array(g.kx)[:, :, 1]
    ky      = Array(g.ky)[:, :, 1]
    mask    = Array(g.mask)[:, :, 1]
    invlap  = Array(g.invlap)[:, :, 1]
    F̂       = Array(g.forcing)[:, :, 1]
    λ       = -(kx.^2 .+ ky.^2) ./ g.Re

    # ---- ETDRK4 coefficients, by contour integrals on 32 points ----
    E  = exp.(dt .* λ)
    E2 = exp.(dt .* λ ./ 2)
    r  = reshape(exp.(im .* π .* ((1:32) .- 0.5) ./ 32), 1, 1, :)
    LR = dt .* λ .+ r
    avg(A) = dropdims(sum(A; dims=3); dims=3) ./ 32

    Q  = dt .* real.(avg((exp.(LR ./ 2) .- 1) ./ LR))
    f1 = dt .* real.(avg((-4 .- LR .+ exp.(LR) .* (4 .- 3LR .+ LR.^2)) ./ LR.^3))
    f2 = dt .* real.(avg((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR.^3))
    f3 = dt .* real.(avg((-4 .- 3LR .- LR.^2 .+ exp.(LR) .* (4 .- LR)) ./ LR.^3))

    # ---- nonlinear term and forcing, dealiased on the physical grid ----
    pad(a) = (P = zeros(ComplexF64, m + 2, M);
              P[1:n + 1, 1:n + 1] .= a[:, 1:n + 1];
              P[1:n + 1, M - n + 1:M] .= a[:, n + 2:2n + 1];
              brfft(P, M))
    trunc(p) = (P = rfft(p) ./ M^2; [P[1:n + 1, 1:n + 1] P[1:n + 1, M - n + 1:M]])

    function Nl(ω̂)
        u  = pad(-im .* ky .* invlap .* ω̂)
        v  = pad(im .* kx .* invlap .* ω̂)
        ωx = pad(im .* kx .* ω̂)
        ωy = pad(im .* ky .* ω̂)
        return mask .* (trunc(.-(u .* ωx .+ v .* ωy)) .+ F̂)
    end

    # ---- steps ----
    ω̂   = mask .* ω̂₀
    out = zeros(ComplexF64, n + 1, 2n + 1, nsteps ÷ every + 1)
    out[:, :, 1] .= ω̂

    for i in 1:nsteps
        Nω = Nl(ω̂)
        â  = E2 .* ω̂ .+ Q .* Nω
        Na = Nl(â)
        b̂  = E2 .* ω̂ .+ Q .* Na
        Nb = Nl(b̂)
        ĉ  = E2 .* â .+ Q .* (2 .* Nb .- Nω)
        Nc = Nl(ĉ)
        ω̂  = E .* ω̂ .+ f1 .* Nω .+ 2 .* f2 .* (Na .+ Nb) .+ f3 .* Nc

        i % every == 0 && (out[:, :, i ÷ every + 1] .= ω̂)
    end

    return out
end

# the state ω̂ translated by ℓ along x, ω(x + ℓ, y)
shift(ω̂::AbstractMatrix, ℓ::Real, g::KFGrid) = ω̂ .* exp.(im .* Array(g.kx)[:, :, 1] .* ℓ)

# Best near-recurrence of the states W, spaced by Δt: the start i, the lag m and the shift ℓ
# minimising ‖W[:, :, i+m] - W(x + ℓ)[:, :, i]‖ relative to ‖W[:, :, i]‖, for lags whose duration
# lies in Trange, the shift on a grid of 4(n+1) values. Starts where the flow is nearly steady, close
# to an equilibrium or a travelling wave, are skipped: there the state changes by less than a
# fraction `motion` over a quarter of the shortest lag, up to a shift.
function recurrence(W::AbstractArray{<:Complex, 3}, g::KFGrid, Δt::Real, Trange; motion::Real=0.3)
    n    = g.n
    lags = ceil(Int, first(Trange) / Δt):floor(Int, last(Trange) / Δt)
    q    = first(lags) ÷ 4

    # ---- weights of the half spectrum, shifts and their phases ----
    w  = [1; fill(2, n)]
    ℓs = range(0, 2π; length=4(n + 1) + 1)[1:end - 1]
    E  = [w[k + 1] * cis(k * ℓ) for ℓ in ℓs, k in 0:n]
    n² = [sum(w .* sum(abs2, W[:, :, i]; dims=2)) for i in axes(W, 3)]

    # largest correlation Re Σ w conj(b̂) â exp(i kx ℓ) over the shifts, and the best shift
    function correlation(i, j)
        S = vec(sum(conj.(W[:, :, j]) .* W[:, :, i]; dims=2))
        c = real.(E * S)
        k = argmax(c)
        return c[k], ℓs[k]
    end

    distance(i, j) = sqrt(max(n²[i] + n²[j] - 2first(correlation(i, j)), 0) / n²[i])
    moving(i)      = distance(i, i + q) > motion

    # ---- best recurrence ----
    best = (Inf, 0, 0, 0.0)
    for i in filter(moving, 1:size(W, 3) - last(lags)), m in lags
        c, ℓ = correlation(i, i + m)
        e    = sqrt(max(n²[i] + n²[i + m] - 2c, 0) / n²[i])
        e < best[1] && (best = (e, i, m, ℓ))
    end

    return best
end

# Initial orbit from the recurrence (i, m, ℓ) of a trajectory: the segment from W[:, :, i], over the
# period T = m Δt, resampled on Ns rescaled times by integrating again with steps of T/(Ns × sub),
# written in the frame drifting with c = -ℓ/T and copied to the device of the grid. Returns the
# Orbit with parameters [log ω, c].
function initial_orbit(W::AbstractArray{<:Complex, 3}, g::KFGrid, Δt::Real, i::Int, m::Int,
                       ℓ::Real; sub::Int=8)
    T = m * Δt
    c = -ℓ / T

    # ---- the segment on Ns equispaced rescaled times ----
    seg = integrate(W[:, :, i], g, T / (g.Ns * sub), g.Ns * sub; every=sub)

    # ---- moving frame: a(x, y, s_l) = ω(x + c t_l, y, t_l) ----
    t = (0:g.Ns - 1) .* (T / g.Ns)
    a = cat([shift(seg[:, :, l], c * t[l], g) for l in 1:g.Ns]...; dims=3)

    return Orbit(KFField(g, g.array(a)), [log(2π / T), c])
end
