# Two-dimensional Kolmogorov flow in space-time form, for ReSolver: the vorticity equation
#
#     ∂t ω + u ∂x ω + v ∂y ω = (1/Re) ∇²ω - kf cos(kf y),     (x, y) ∈ [0, 2π)² periodic,
#
# with velocity u = -∂y ψ, v = ∂x ψ from the streamfunction, ∇²ψ = ω, with the conventions of
# OpenKolmogorovFlow.jl (and of Chandler & Kerswell 2013).
#
# The method and the notation are those of the paper "Space-time Newton–Krylov searches for unstable
# periodic orbits of two-dimensional Kolmogorov flow" (D. Lasagna, in preparation) and of the page
# docs/src/examples/kolmogorov_flow.md. The names here differ in three places: the vorticity η of
# the paper is ω here (the frequency is ReSolver.frequency), the cutoff K of the paper is n, and
# its forcing wavenumber n is kf. The paper uses Re = 40, kf = 4 and n = 33.
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
# a discrete symmetry, optional and not used in the paper: an orbit may repeat after a period up to
# such a shift ℓy,
# ω(x, y, t + T) = ω(x + ℓx, y + ℓy, t). The grid keyword `yshift` = ℓy makes the space-time field
# periodic up to that shift, â(kx, ky, s + 2π) = exp(i ky ℓy) â(kx, ky, s): the derivative in s acts
# on the periodic part exp(-i φ s) â, φ = ky ℓy/2π, as a multiplication by i(n + φ) on its Fourier
# modes n. Only ∂s and the preconditioner change; ∂s stays exactly skew-adjoint.
#
# The model runs on the CPU or on a GPU: the arrays of the grid and of the fields are built with the
# constructor `array` given to the grid, `Array` by default or e.g. `CuArray` with CUDA.jl loaded,
# the FFTs go through AbstractFFTs, and the operators use broadcasting and views only, no scalar
# indexing. They work in double or single precision, the keyword `precision` of the grid. The
# initial guess is built on the CPU in double precision and copied to the device.
#
# The file provides what a ReSolver System needs, and the pieces to build an initial guess:
#
#     KFGrid(n, Ns; Re, kf, yshift,    grid: wavenumbers, mask, FFT plans, on the device
#            array, precision)
#     KFField(g)                       space-time Fourier field, zero
#     ddx!(out, ω), dds!(out, ω)       derivatives along x and in rescaled time s
#     KFNonlinear, KFLinearised        N(ω), its linearisation and adjoint
#     linearise!(op, ω)                linearisation point of KFLinearised
#     KFPreconditioner                 diagonal or factorised preconditioner, kolmogorov_flow.md
#     physical(ω, l)                   the vorticity at the rescaled time s_l on the physical grid
#     integrate, recurrence            time integration (ETDRK4) and near-recurrences
#     initial_orbit, coefficients      an initial orbit from a recurrence, or from a seed

using FFTW
using LinearAlgebra
using ReSolver

import ReSolver: precondition!, precondition_adjoint!


# ============================================================================ #
# Grid                                                                         #
# ============================================================================ #

# active and storage cutoffs, and points of the physical grid, as in OpenKolmogorovFlow.jl
up_dealias_size(n::Int) = n + n >> 1

"""
    KFGrid(n, Ns; Re=40, kf=4, yshift=0, array=Array, precision=:double)

Grid of the space-time Kolmogorov flow: the active modes 0 ≤ kx ≤ n, |ky| ≤ n, the physical grid
of (2m+2)² points, m = n + n÷2, which dealiases the products by the 3/2 rule, and `Ns` instants in
rescaled time, `Ns` odd. `array` builds the arrays on the device, e.g. `CuArray`, and `precision`
is `:double` or `:single`. `yshift` makes the loops periodic up to a translation along y by that
amount, a multiple of 2π/kf; zero for periodic loops.
"""
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
    viscous::R       # viscous term (1/Re) ∇², -(kx² + ky²)/Re
    forcing::C       # coefficients of the forcing -kf cos(kf y)
    physfwd::FP      # real FFT over (x, y), for every s, physical grid → padded coefficients
    physinv::IP      # its unnormalised inverse
     timefwd::FS     # FFT along s of the active coefficients, in place
     timeinv::IS     # its unnormalised inverse, in place
      array::A       # constructor of the arrays on the device, Array or e.g. CuArray

    function KFGrid(n::Int,
                   Ns::Int;
                   Re::Real=40,
                   kf::Int=4,
               yshift::Real=0,
                array=Array,
            precision::Symbol=:double)

        # ---- input checks ----
        isodd(Ns) ||
            throw(ArgumentError("Ns must be odd"))
        kf <= n ||
            throw(ArgumentError("the forcing wavenumber must be active, kf ≤ n"))
        precision in (:double, :single) ||
            throw(ArgumentError("precision must be :double or :single"))

        # ---- sizes and wavenumbers, in the floating-point type of the precision ----
        T  = precision === :double ? Float64 : Float32
        m  = up_dealias_size(n)
        M  = 2m + 2
        kx = reshape(collect(T, 0:n), :, 1, 1)
        ky = reshape(T[0:n; -n:-1], 1, :, 1)
        s  = reshape(T.(fftfreq(Ns, Ns)), 1, 1, :)

        # ---- shifted temporal wavenumbers and phases of the shift along y ----
        φ     = ky .* T(yshift / 2π)
        σ     = s .+ φ
        twist = cis.(φ .* reshape(T.((0:Ns - 1) .* (2π / Ns)), 1, 1, :))

        # ---- mask of the mean mode, inverse Laplacian and viscous term ----
        k²      = kx.^2 .+ ky.^2
        mask    = @. T(k² != 0)
        invlap  = @. ifelse(k² == 0, zero(T), -1 / k²)
        viscous = @. -k² / T(Re)

        # ---- forcing -kf cos(kf y): coefficients -kf/2 at (0, ±kf), at every s ----
        forcing = zeros(Complex{T}, n + 1, 2n + 1, Ns)
        forcing[1, kf + 1, :]      .= -kf / 2
        forcing[1, 2n + 2 - kf, :] .= -kf / 2

        # ---- FFT plans on the device ----
        u = array(zeros(T, M, M, Ns))
        P = array(zeros(Complex{T}, m + 2, M, Ns))
        a = array(zeros(Complex{T}, n + 1, 2n + 1, Ns))

        physfwd = plan_rfft(u, (1, 2))
        physinv = plan_brfft(P, M, (1, 2))
        timefwd = plan_fft!(a, 3)
        timeinv = plan_bfft!(a, 3)

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
                                  array(viscous),
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
function _pad!(P::AbstractArray{<:Complex, 3}, a::AbstractArray{<:Complex, 3}, g::KFGrid)
    n, M = g.n, g.M

    fill!(P, 0)
    view(P, 1:n + 1, 1:n + 1, :)     .= view(a, :, 1:n + 1, :)
    view(P, 1:n + 1, M - n + 1:M, :) .= view(a, :, n + 2:2n + 1, :)

    return P
end

function _truncate!(a::AbstractArray{<:Complex, 3}, P::AbstractArray{<:Complex, 3}, g::KFGrid)
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
times, an (n+1) × (2n+1) × Ns complex array on the device and in the precision of the grid.
"""
struct KFField{T, G<:KFGrid, D} <: AbstractArray{T, 3}
    data::D # active coefficients at the rescaled times
       g::G # grid, shared

    KFField(g::G, data::D=zero(g.forcing)) where {G<:KFGrid, D} = new{eltype(D), G, D}(data, g)
end

# ---- array interface ----
Base.size(u::KFField)                    = size(u.data)
Base.getindex(u::KFField, I::Int...)     = u.data[I...]
Base.setindex!(u::KFField, v, I::Int...) = (u.data[I...] = v)
Base.IndexStyle(::Type{<:KFField})       = IndexLinear()

Base.similar(u::KFField) = KFField(u.g, similar(u.data))
Base.copy(u::KFField)    = KFField(u.g, copy(u.data))

# Mean over space and rescaled time of the products of the fields: by Parseval, the sum over the full
# spectrum, where each stored mode with kx > 0 stands for itself and its conjugate, kx < 0. Both sums
# are dot products of vectors, by BLAS on the CPU and CUBLAS on the GPU: of all the coefficients,
# and of those with kx = 0, a strided vector
function LinearAlgebra.dot(u::KFField, v::KFField)
    n = u.g.n

    all = real(dot(vec(u.data), vec(v.data)))
    kx₀ = real(dot(view(reshape(u.data, n + 1, :), 1, :), view(reshape(v.data, n + 1, :), 1, :)))

    return (2all - kx₀) / u.g.Ns
end

LinearAlgebra.norm(u::KFField) = sqrt(dot(u, u))

# ---- broadcasting keeps the field type, and runs on the arrays of the device ----
Base.BroadcastStyle(::Type{<:KFField}) = Broadcast.ArrayStyle{KFField}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{KFField}}, ::Type{<:Complex}) =
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
    P = zeros(eltype(a), g.m + 2, g.M)

    P[1:g.n + 1, 1:g.n + 1]           .= a[:, 1:g.n + 1]
    P[1:g.n + 1, g.M - g.n + 1:g.M]   .= a[:, g.n + 2:2g.n + 1]

    return brfft(P, g.M)
end


# ============================================================================ #
# Derivatives                                                                  #
# ============================================================================ #

# out = ∂x u, a multiplication of the coefficients
ddx!(out::KFField, u::KFField) = (out.data .= im .* u.g.kx .* u.data; out)

# out = ∂s u, by FFTs along s of the periodic part exp(-i φ s) u, in place in out
function dds!(out::KFField, u::KFField)
    g = u.g

    out.data .= conj.(g.twist) .* u.data
    g.timefwd * out.data
    out.data .*= im .* g.σ ./ g.Ns
    g.timeinv * out.data
    out.data .*= g.twist

    return out
end


# ============================================================================ #
# Operators                                                                    #
# ============================================================================ #

# Work arrays of the operators: four fields on the physical grid, four arrays of padded coefficients
# and two of active coefficients, on the device of the grid.
function _work(g::KFGrid)
    T      = real(eltype(g.forcing))
    phys   = ntuple(i -> g.array(zeros(T, g.M, g.M, g.Ns)), 4)
    padded = ntuple(i -> g.array(zeros(Complex{T}, g.m + 2, g.M, g.Ns)), 4)
    active = ntuple(i -> zero(g.forcing), 2)

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
    out.data .= g.mask .* (active[2] .+ g.viscous .* ω.data .+ g.forcing)

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
        base = ntuple(i -> g.array(zeros(real(eltype(g.forcing)), g.M, g.M, g.Ns)), 4)
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
    out.data .= g.mask .* (out.data .+ g.viscous .* v.data)

    return out
end


# ============================================================================ #
# Preconditioner                                                               #
# ============================================================================ #

# Preconditioners built about an orbit x₀, of frequency ω₀ and drift speed c₀. Two are
# diagonal in space-time Fourier, and divide the coefficient of the mode (kx, ky, m) by
#
#     kind = :jacobian    λ = i(ω₀ m - c₀ kx) + (kx² + ky²)/Re, the linear space-time operator about
#                         the state of zero vorticity, J_t + L₀ in the notation of the paper
#     kind = :linear      1 + |λ|, its size, positive: the metric of L-BFGS
#
# with 1 on the mean mode, which is excluded. The third is the factorised preconditioner of the paper
# (docs/src/examples/kolmogorov_flow.md),
#
#     kind = :factorised  B = (J_t + σ)(σ + J_x)/σ, with the pieces of the field block of the
#                         Jacobian about x₀: the time derivative J_t = ω₀ ∂s and the spatial operator
#                         J_x = L₀ + J_conv, L₀ = J_drift + J_visc, J_drift = -c₀ ∂x, J_visc the
#                         viscous term and J_conv the convective term. B⁻¹ divides the temporal mode
#                         m by 1 + i ω₀ m/σ, then solves the blocks σ + J_x(s_l) at all the times at
#                         once by `inner` steps of GMRES from zero, right preconditioned by their
#                         diagonal part σ + L₀ = σ - i c₀ kx + (kx² + ky²)/Re
#
# With m shifted by φ = ky ℓy/2π for an orbit shifted along y. The paper uses σ = min(Ns/T, 2.5) and
# inner = 10, with NewtonHookstep(flexible=true); the defaults σ = 3 and inner = 5 are not these
# settings. The inexact inner solves make the factorised B⁻¹ slightly nonlinear: flexible GMRES
# forms the step from the preconditioned vectors and stays consistent, and B⁻¹ has no adjoint for
# L-BFGS. The parameters are divided by ‖ω₀ ∂s ω₀‖ and ‖∂x ω₀‖, the norms of their columns of the
# Jacobian. Built about x₀, the preconditioner is rebuilt about the current orbit at every Newton
# iteration of the hookstep, by ReSolver.relinearise!, below.
"""
    KFPreconditioner(x₀::Orbit; kind=:jacobian, σ=3.0, inner=5)

Preconditioner of the Kolmogorov flow built about the orbit `x₀`, for the `B` of a `System`.
`kind=:jacobian` and `kind=:linear` are diagonal in space-time Fourier; `kind=:factorised` is the
factorised preconditioner of the paper, with shift `σ` and `inner` steps of GMRES on the blocks at
the instants, for the hookstep with `NewtonHookstep(flexible=true)`. The paper uses
`σ=min(Ns / T, 2.5)` and `inner=10`.
"""
mutable struct KFPreconditioner{C, L, W}
     kind::Symbol            # :jacobian, :linear or :factorised
        β::C                 # divisors of the temporal Fourier modes, (n+1) × (2n+1) × Ns
        d::C                 # :factorised, diagonal part of the blocks, (n+1) × (2n+1) × 1
        s::Vector{Float64}   # parameter scales
      lin::L                 # :factorised, operator linearised about x₀; nothing otherwise
        σ::Float64           # :factorised, shift
    inner::Int               # :factorised, steps of GMRES on the blocks
       c₀::Float64           # drift speed of the orbit it is built about
     work::W                 # :factorised, inner products weights and work fields; nothing otherwise

    function KFPreconditioner(x₀::Orbit; kind::Symbol=:jacobian, σ::Real=3.0, inner::Int=5)

        # ---- input checks ----
        kind in (:jacobian, :linear, :factorised) ||
            throw(ArgumentError("kind must be :jacobian, :linear or :factorised"))

        a₀   = x₀.a
        g    = a₀.g
        T    = eltype(a₀.data)
        c₀   = length(x₀.p) > 1 ? x₀.p[2] : 0.0
        β, d = _divisors(kind, x₀, σ)

        # ---- the factorised kind: the linearised operator, inner product weights, work fields ----
        lin  = nothing
        work = nothing
        if kind === :factorised
            lin  = linearise!(KFLinearised(g), a₀)
            work = (weight = g.array(reshape(real(T)[1; fill(2, g.n)], :, 1, 1)),
                    fields = [KFField(g) for _ in 1:inner + 4])
        end

        return new{typeof(β), typeof(lin), typeof(work)}(kind, β, d, _scales(x₀), lin, σ, inner, c₀,
                                                         work)
    end
end

# the divisors of the temporal modes, in the precision of the field and 1 on the mean mode, and the
# diagonal part of the blocks, with the frequency and the drift speed of the orbit x
function _divisors(kind::Symbol, x::Orbit, σ::Real)
    g = x.a.g
    T = eltype(x.a.data)
    ω = ReSolver.frequency(x)
    c = length(x.p) > 1 ? x.p[2] : 0.0

    λ = @. im * (ω * g.σ - c * g.kx) - g.viscous
    b = kind === :jacobian ? λ :
        kind === :linear   ? @.(1 + abs(λ) + 0im) :
                             @.(1 + im * ω * g.σ / σ + zero(λ))
    β = T.(@. ifelse(g.mask == 0, one(b), b))
    d = T.(@. σ - im * c * g.kx - g.viscous)

    return β, d
end

# the scales of the parameters: the norms of their columns of the Jacobian, ‖ω ∂s a‖ and ‖∂x a‖
function _scales(x::Orbit)
    ∂a = similar(x.a)
    s  = [norm(ReSolver.frequency(x) .* dds!(∂a, x.a))]
    length(x.p) > 1 && push!(s, norm(ddx!(∂a, x.a)))

    return s
end

# The preconditioner rebuilt about the orbit x, in place, at every Newton iteration of the hookstep:
# the divisors and the diagonal part with the frequency and the drift speed of x, the operator of the
# blocks linearised about its field, and the parameter scales; the kind, σ and the inner steps stay
function ReSolver.relinearise!(B::KFPreconditioner, x::Orbit)
    β, d = _divisors(B.kind, x, B.σ)

    B.β .= β
    B.d .= d
    B.s .= _scales(x)
    B.c₀ = length(x.p) > 1 ? x.p[2] : 0.0
    B.kind === :factorised && linearise!(B.lin, x.a)

    return B
end

# The temporal modes are divided on the periodic part exp(-i φ s) of the field, then the factorised
# preconditioner solves its blocks; B⁻⁺ of the diagonal kinds divides by the conjugate multipliers
function precondition!(out::Orbit, B::KFPreconditioner, p::Orbit)
    g = p.a.g

    out.a.data .= conj.(g.twist) .* p.a.data
    g.timefwd * out.a.data
    out.a.data .*= g.mask ./ (B.β .* g.Ns)
    g.timeinv * out.a.data
    out.a.data .*= g.twist

    B.kind === :factorised && _blocks!(out.a, B)

    out.p .= p.p ./ B.s

    return out
end

function precondition_adjoint!(out::Orbit, B::KFPreconditioner, p::Orbit)
    B.kind === :factorised &&
        throw(ArgumentError("the factorised preconditioner has no adjoint: use it with the hookstep"))

    g = p.a.g

    out.a.data .= conj.(g.twist) .* p.a.data
    g.timefwd * out.a.data
    out.a.data .*= g.mask ./ (conj.(B.β) .* g.Ns)
    g.timeinv * out.a.data
    out.a.data .*= g.twist

    out.p .= p.p ./ B.s

    return out
end

# The preconditioner applied forward, the metric of the trust region of the hookstep with metric=:operator: for the
# factorised kind B p = (J_t + σ)(σ + J_x) p/σ, the blocks applied at all the times, then the temporal modes
# multiplied by 1 + i ω₀ m/σ on the periodic part; for the diagonal kinds the multipliers; the scales on the
# parameters
function ReSolver.metric!(out::Orbit, B::KFPreconditioner, p::Orbit)
    g  = p.a.g
    R  = real(eltype(p.a.data))
    σ  = R(B.σ)
    c₀ = R(B.c₀)

    # ---- the blocks σ + J_x, for the factorised kind ----
    if B.kind === :factorised
        tmp = B.work.fields[end]
        B.lin(out.a, p.a)
        out.a.data .= σ .* p.a.data .- c₀ .* ddx!(tmp, p.a).data .- out.a.data
    else
        out.a.data .= p.a.data
    end

    # ---- the temporal modes, on the periodic part ----
    out.a.data .*= conj.(g.twist)
    g.timefwd * out.a.data
    out.a.data .*= g.mask .* B.β ./ g.Ns
    g.timeinv * out.a.data
    out.a.data .*= g.twist

    out.p .= p.p .* B.s

    return out
end

# The blocks σ + J_u,drift + J_u,visc + J_u,conv(s_l) of the factorised preconditioner, solved in
# place at all the times at once by B.inner steps of GMRES from zero, right preconditioned by their
# diagonal part B.d: each time has
# its own inner products, Hessenberg matrix and least-squares problem, while every inner step applies
# the operator to all the times together.
function _blocks!(z::KFField, B::KFPreconditioner)
    g  = z.g
    R  = real(eltype(z.data))
    k  = B.inner
    σ  = R(B.σ)
    c₀ = R(B.c₀)

    Q   = B.work.fields[1:k + 1] # Krylov vectors
    f   = B.work.fields[k + 2]   # right-hand side
    w   = B.work.fields[k + 3]   # operator output
    tmp = B.work.fields[k + 4]   # derivative along x

    # inner products at each time, the modes kx > 0 counted twice for their conjugates, and a vector
    # of a number per time as a 1 × 1 × Ns array on the device
    dots(u, v) = Float64.(Array(vec(sum(B.work.weight .* real.(conj.(u) .* v); dims=(1, 2)))))
    bytime(h)  = reshape(g.array(R.(h)), 1, 1, :)

    # ---- the right-hand side, normalised at each time ----
    f.data .= z.data
    β₀ = sqrt.(dots(f.data, f.data))
    H  = zeros(k + 1, k, g.Ns)
    Q[1].data .= f.data ./ bytime(max.(β₀, eps()))

    # ---- Arnoldi, two passes of Gram–Schmidt; z holds D⁻¹ q_j ----
    for j in 1:k
        z.data .= Q[j].data ./ B.d
        B.lin(w, z)
        w.data .= σ .* z.data .- c₀ .* ddx!(tmp, z).data .- w.data
        for _ in 1:2, i in 1:j
            h          = dots(Q[i].data, w.data)
            w.data   .-= bytime(h) .* Q[i].data
            H[i, j, :] .+= h
        end
        H[j + 1, j, :] .= sqrt.(dots(w.data, w.data))
        Q[j + 1].data  .= w.data ./ bytime(max.(H[j + 1, j, :], eps()))
    end

    # ---- least squares at each time, and the solution D⁻¹ Q y ----
    Y = zeros(k, g.Ns)
    for l in 1:g.Ns
        e₁       = zeros(k + 1)
        e₁[1]    = β₀[l]
        Y[:, l] .= H[:, :, l] \ e₁
    end
    fill!(z.data, 0)
    for j in 1:k
        z.data .+= bytime(Y[j, :]) .* Q[j].data
    end
    z.data ./= B.d

    return z
end


# ============================================================================ #
# Time integration and near-recurrences, for initial guesses, on the CPU       #
# ============================================================================ #

# Integrate a state ω̂₀ (the (n+1) × (2n+1) active coefficients) for nsteps steps of size dt with
# ETDRK4 (Kassam & Trefethen 2005), products dealiased on the physical grid, returning the states
# every `every` steps as the slices of an (n+1) × (2n+1) × (nsteps ÷ every + 1) array.
function integrate(ω̂₀::AbstractMatrix, g::KFGrid, dt::Real, nsteps::Int; every::Int=1)
    n, m, M = g.n, g.m, g.M
    kx      = Float64.(Array(g.kx)[:, :, 1])
    ky      = Float64.(Array(g.ky)[:, :, 1])
    mask    = Float64.(Array(g.mask)[:, :, 1])
    invlap  = Float64.(Array(g.invlap)[:, :, 1])
    F̂       = ComplexF64.(Array(g.forcing)[:, :, 1])
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

    # ---- transforms of one state, planned once and on one thread, since they are small ----
    P   = zeros(ComplexF64, m + 2, M)
    fwd = plan_rfft(zeros(M, M); num_threads=1)
    bwd = plan_brfft(P, M; num_threads=1)

    # ---- nonlinear term and forcing, dealiased on the physical grid ----
    function pad(a)
        fill!(P, 0)
        P[1:n + 1, 1:n + 1]     .= a[:, 1:n + 1]
        P[1:n + 1, M - n + 1:M] .= a[:, n + 2:2n + 1]
        return bwd * P
    end

    function trunc(p)
        p̂ = (fwd * p) ./ M^2
        return [p̂[1:n + 1, 1:n + 1] p̂[1:n + 1, M - n + 1:M]]
    end

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
shift(ω̂::AbstractMatrix, ℓ::Real, g::KFGrid) = ω̂ .* exp.(im .* Float64.(Array(g.kx)[:, :, 1]) .* ℓ)

# Best near-recurrence of the states W, spaced by Δt: the start i, the lag m and the shifts ℓx, ℓy
# minimising ‖W[:, :, i+m] - W(x + ℓx, y + ℓy)[:, :, i]‖ relative to ‖W[:, :, i]‖, for lags whose
# duration lies in Trange: ℓx on a grid of 4(n+1) values, ℓy a multiple of the forcing wavelength
# 2π/kf, a discrete symmetry. Starts where the flow is nearly steady, close to an equilibrium or a
# travelling wave, are skipped: there the state changes by less than a fraction `motion` over a
# quarter of the shortest lag, up to a shift.
function recurrence(W::AbstractArray{<:Complex, 3}, g::KFGrid, Δt::Real, Trange; motion::Real=0.3)
    n    = g.n
    lags = ceil(Int, first(Trange) / Δt):floor(Int, last(Trange) / Δt)
    q    = first(lags) ÷ 4

    # ---- weights of the half spectrum, shifts and their phases ----
    w   = [1; fill(2, n)]
    ky  = Float64.(Array(g.ky)[1, :, 1])
    ℓxs = range(0, 2π; length=4(n + 1) + 1)[1:end - 1]
    ℓys = (0:g.kf - 1) .* (2π / g.kf)
    E   = [w[k + 1] * cis(k * ℓ) for ℓ in ℓxs, k in 0:n]
    Y   = [cis.(ky .* ℓ) for ℓ in ℓys]
    n²  = [sum(w .* sum(abs2, W[:, :, i]; dims=2)) for i in axes(W, 3)]

    # largest correlation Re Σ w conj(b̂) â exp(i (kx ℓx + ky ℓy)) over the shifts, and the best ones
    function correlation(i, j)
        # (the name top, not best, keeps the closure from overwriting the result of the outer loop)
        P   = conj.(W[:, :, j]) .* W[:, :, i]
        top = (-Inf, 0.0, 0.0)
        for (ℓy, y) in zip(ℓys, Y)
            c = real.(E * (P * y))
            k = argmax(c)
            c[k] > top[1] && (top = (c[k], ℓxs[k], ℓy))
        end
        return top
    end

    distance(i, j) = sqrt(max(n²[i] + n²[j] - 2first(correlation(i, j)), 0) / n²[i])
    moving(i)      = distance(i, i + q) > motion

    # ---- best recurrence ----
    best = (Inf, 0, 0, 0.0, 0.0)
    for i in filter(moving, 1:size(W, 3) - last(lags)), m in lags
        c, ℓx, ℓy = correlation(i, i + m)
        e         = sqrt(max(n²[i] + n²[i + m] - 2c, 0) / n²[i])
        e < best[1] && (best = (e, i, m, ℓx, ℓy))
    end

    return best
end

# The coefficients of a vorticity field given as by OpenKolmogorovFlow.jl, Ω[ky, kx], ky in FFT order
# along the first dimension and kx ≥ 0 along the second, on the active modes of the resolution n,
# 0 ≤ kx ≤ n and |ky| ≤ n, in the layout of the model, (n+1) × (2n+1); the modes Ω lacks are zero,
# and so is the mean.
function coefficients(Ω::AbstractMatrix, n::Int)
    K = size(Ω, 1)
    ω̂ = zeros(ComplexF64, n + 1, 2n + 1)
    for kx in 0:min(n, size(Ω, 2) - 1), ky in -n:n
        abs(ky) < K ÷ 2 || continue
        ω̂[kx + 1, ky >= 0 ? ky + 1 : 2n + 2 + ky] = Ω[mod(ky, K) + 1, kx + 1]
    end
    ω̂[1, 1] = 0
    return ω̂
end

# Initial orbit from the recurrence (i, m, ℓx, ℓy) of a trajectory, on a grid g whose y-shift is ℓy:
# the segment from W[:, :, i], over the period T = m Δt, resampled on Ns rescaled times by
# integrating again with steps of T/(Ns × sub), written in the frame drifting with c = -ℓx/T and
# copied to the device of the grid. Returns the Orbit with parameters [log ω, c].
function initial_orbit(W::AbstractArray{<:Complex, 3}, g::KFGrid, Δt::Real, i::Int, m::Int,
                       ℓx::Real; sub::Int=8)
    T = m * Δt
    c = -ℓx / T

    # ---- the segment on Ns equispaced rescaled times ----
    seg = integrate(W[:, :, i], g, T / (g.Ns * sub), g.Ns * sub; every=sub)

    # ---- moving frame: a(x, y, s_l) = ω(x + c t_l, y, t_l) ----
    t = (0:g.Ns - 1) .* (T / g.Ns)
    a = cat([shift(seg[:, :, l], c * t[l], g) for l in 1:g.Ns]...; dims=3)

    return Orbit(KFField(g, g.array(eltype(g.forcing).(a))), [log(2π / T), c])
end

# Initial orbit from the seeds of a multiple-shooting solution, on a grid g whose y-shift is ℓy: the
# states W[k] at the starts of N segments of equal duration T/N, each integrated over its segment
# with steps of T/(Ns sub), sub made a multiple of N so that the segments start on steps, and
# sampled on the Ns rescaled times, in the frame drifting with c = -ℓx/T. One seed is a trajectory
# over the whole period. Returns the Orbit with parameters [log ω, c].
function initial_orbit(W::AbstractVector{<:AbstractMatrix}, g::KFGrid, T::Real, ℓx::Real; sub::Int=8)
    N   = length(W)
    Ns  = g.Ns
    c   = -ℓx / T
    sub = N * cld(sub, N)
    L   = Ns * sub ÷ N
    dt  = T / (Ns * sub)

    # ---- each segment from its seed; the states on the multiples of sub, in the moving frame ----
    a = zeros(ComplexF64, g.n + 1, 2g.n + 1, Ns)
    for k in 1:N
        seg = integrate(W[k], g, dt, L; every=1)
        for j in 0:L - 1
            i = (k - 1) * L + j
            i % sub == 0 || continue
            l = i ÷ sub + 1
            a[:, :, l] .= shift(seg[:, :, j + 1], c * (l - 1) * T / Ns, g)
        end
    end

    return Orbit(KFField(g, g.array(eltype(g.forcing).(a))), [log(2π / T), c])
end
