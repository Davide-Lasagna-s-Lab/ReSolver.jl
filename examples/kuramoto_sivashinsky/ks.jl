# Kuramoto–Sivashinsky equation in space-time form, for ReSolver.
#
#     ∂t u = -u ∂x u - ∂xx u - ∂xxxx u,     x ∈ [0, L) periodic.
#
# A space-time field is a KSField: real values u[j, l] on the grid x_j = (j - 1) L/Nx,
# s_l = 2π (l - 1)/Ns of space and rescaled time, with Nx and Ns odd. Its transform, a
# KSTransformedField, holds the Fourier coefficients: real FFT along x, full FFT along s. Derivatives
# and the linear operator are diagonal there; products are computed on the grid. The transforms are
# planned once, in a KSFFT owned by the grid.
#
# The inner product is the mean over the grid points, the discretisation of (1/2πL) ∫∫ u v dx ds, so
# that norms do not depend on the resolution. With odd sizes the spectral derivatives are exactly
# skew-adjoint in it, and the adjoint operator is exact.
#
# The file provides what a ReSolver System needs, and the pieces to build an initial guess:
#
#     KSGrid(L, Nx, Ns)                grid, wavenumbers and FFT plans
#     KSField(g)                       space-time field, zero
#     ddx!(out, u), dds!(out, u)       derivatives along x and in rescaled time s
#     KSNonlinear, KSLinearised        N(u), its linearisation and adjoint
#     linearise!(op, u)                linearisation point of KSLinearised
#     KSPreconditioner                 diagonal Fourier preconditioner
#     demean!(u)                       projection on the fields of zero mean
#     odd!(u)                          projection on the odd fields, u(-x) = -u(x)
#     resample(u, g)                   spectral interpolation to another grid
#     integrate, recurrence            time integration (ETDRK4) and near-recurrences

using FFTW
using LinearAlgebra
using ReSolver

import ReSolver: precondition!, precondition_adjoint!


# ============================================================================ #
# Grid and transforms                                                          #
# ============================================================================ #

# Planned transforms between space-time values and Fourier coefficients, with a scratch array of
# coefficients for the derivatives.
struct KSFFT{F, I}
    forward::F                  # real FFT along x, full FFT along s
    inverse::I                  # its inverse, which overwrites its input
    scratch::Matrix{ComplexF64} # coefficients, used within one derivative

    function KSFFT(Nx::Int, Ns::Int)
        u = zeros(Nx, Ns)
        û = rfft(u, (1, 2))

        forward = plan_rfft(u, (1, 2))
        inverse = plan_irfft(û, Nx, (1, 2))

        return new{typeof(forward), typeof(inverse)}(forward, inverse, similar(û))
    end
end

struct KSGrid{F<:KSFFT}
     L::Float64         # domain length
    Nx::Int             # points along x, odd
    Ns::Int             # points in rescaled time s, odd
     k::Vector{Float64} # wavenumbers along x, 2π/L × (0, 1, …, (Nx-1)/2)
     n::Vector{Float64} # signed temporal wavenumbers along s, in FFT order
     λ::Vector{Float64} # linear operator of KS on each wavenumber, k² - k⁴
   fft::F               # planned transforms

    function KSGrid(L::Real, Nx::Int, Ns::Int)

        # ---- input checks ----
        isodd(Nx) && isodd(Ns) ||
            throw(ArgumentError("Nx and Ns must be odd"))

        # ---- wavenumbers and linear operator ----
        k = collect((2π / L) .* (0:(Nx - 1) ÷ 2))
        n = collect(fftfreq(Ns, Ns))
        λ = k.^2 .- k.^4

        # ---- transforms ----
        fft = KSFFT(Nx, Ns)

        return new{typeof(fft)}(L, Nx, Ns, k, n, λ, fft)
    end
end

# grid points along x
xpoints(g::KSGrid) = (0:g.Nx - 1) .* (g.L / g.Nx)


# ============================================================================ #
# Fields                                                                       #
# ============================================================================ #

"""
    KSField(g, data=zeros(g.Nx, g.Ns))

Space-time field on the grid `g`: values at the points (x_j, s_l). Behaves as a real matrix.
"""
struct KSField{G<:KSGrid} <: AbstractMatrix{Float64}
    data::Matrix{Float64} # values on the grid
       g::G               # grid, shared

    KSField(g::G, data::Matrix{Float64}=zeros(g.Nx, g.Ns)) where {G<:KSGrid} = new{G}(data, g)
end

"""
    KSTransformedField(g)

Fourier coefficients of a space-time field: real FFT along x, full FFT along s.
"""
struct KSTransformedField{G<:KSGrid} <: AbstractMatrix{ComplexF64}
    data::Matrix{ComplexF64} # coefficients, (Nx+1)/2 × Ns
       g::G                  # grid, shared

    KSTransformedField(g::G) where {G<:KSGrid} = new{G}(zeros(ComplexF64, length(g.k), g.Ns), g)
end

# ---- array interface ----
for F in (:KSField, :KSTransformedField)
    @eval begin
        Base.size(u::$F)                            = size(u.data)
        Base.getindex(u::$F, I::Int...)             = u.data[I...]
        Base.setindex!(u::$F, v, I::Int...)         = (u.data[I...] = v)
        Base.IndexStyle(::Type{<:$F})               = IndexLinear()
    end
end

Base.similar(u::KSField)            = KSField(u.g)
Base.similar(u::KSTransformedField) = KSTransformedField(u.g)
Base.copy(u::KSField)               = KSField(u.g, copy(u.data))

# mean over the grid points of the products: the discretisation of (1/2πL) ∫∫ u v dx ds
LinearAlgebra.dot(u::KSField, v::KSField) = dot(u.data, v.data) / length(u.data)
LinearAlgebra.norm(u::KSField)            = sqrt(dot(u, u))

# ---- broadcasting keeps the field type ----
Base.BroadcastStyle(::Type{<:KSField}) = Broadcast.ArrayStyle{KSField}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{KSField}}, ::Type{Float64}) =
    KSField(_find_field(bc).g)

_find_field(bc::Broadcast.Broadcasted) = _find_field(bc.args)
_find_field(args::Tuple)               = _find_field(_find_field(args[1]), Base.tail(args))
_find_field(u::KSField, rest)          = u
_find_field(::Any, rest)               = _find_field(rest)
_find_field(u)                         = u

# ---- transforms; the inverse overwrites its input ----
transform!(û::KSTransformedField, u::KSField) = (mul!(û.data, u.g.fft.forward, u.data); û)
transform!(u::KSField, û::KSTransformedField) = (mul!(u.data, u.g.fft.inverse, û.data); u)


# ============================================================================ #
# Derivatives                                                                  #
# ============================================================================ #

# out = ∂x^p u, through the scratch coefficients of the grid
function ddx!(out::KSField, u::KSField, p::Int=1)
    g = u.g
    û = g.fft.scratch

    mul!(û, g.fft.forward, u.data)
    û .*= (im .* g.k).^p
    mul!(out.data, g.fft.inverse, û)

    return out
end

# out = ∂s u, through the scratch coefficients of the grid
function dds!(out::KSField, u::KSField)
    g = u.g
    û = g.fft.scratch

    mul!(û, g.fft.forward, u.data)
    û .*= transpose(im .* g.n)
    mul!(out.data, g.fft.inverse, û)

    return out
end


# ============================================================================ #
# Operators                                                                    #
# ============================================================================ #

# The operators are those of KS. Their output has zero mean for every input, and the search removes
# the mean with the projection demean! (see Projections below).

# N(u) = (k² - k⁴) û - ½ ∂x(u²)
struct KSNonlinear{G<:KSGrid}
     g::G                     # grid
     w::KSField{G}            # scratch, u²
     û::KSTransformedField{G} # scratch, coefficients of u
     ŵ::KSTransformedField{G} # scratch, coefficients of u²

    KSNonlinear(g::G) where {G<:KSGrid} =
        new{G}(g, KSField(g), KSTransformedField(g), KSTransformedField(g))
end

function (op::KSNonlinear)(out::KSField, u::KSField)
    g = op.g

    # ---- coefficients of u and u² ----
    op.w.data .= u.data.^2

    transform!(op.û, u)
    transform!(op.ŵ, op.w)

    # ---- linear part and advection ----
    op.û.data .= g.λ .* op.û.data .- (im .* g.k ./ 2) .* op.ŵ.data

    return transform!(out, op.û)
end

# linearisation about U, L v = (k² - k⁴) v - ∂x(U v), and its adjoint L⁺ w = (k² - k⁴) w + U ∂x w
struct KSLinearised{G<:KSGrid}
          g::G                     # grid
          U::KSField{G}            # linearisation point, set by linearise!
    adjoint::Bool                  # false: L, true: L⁺
          w::KSField{G}            # scratch
          v̂::KSTransformedField{G} # scratch, coefficients of v
          ŵ::KSTransformedField{G} # scratch, coefficients of the advection term

    KSLinearised(g::G; adjoint::Bool=false) where {G<:KSGrid} =
        new{G}(g, KSField(g), adjoint, KSField(g), KSTransformedField(g), KSTransformedField(g))
end

function (op::KSLinearised)(out::KSField, v::KSField)
    g = op.g

    # ---- coefficients of the argument ----
    transform!(op.v̂, v)

    # ---- advection about U, on the grid: U ∂x w for L⁺, with ∂x w from the coefficients of w;
    #      U v for L, differentiated below ----
    if op.adjoint
        op.ŵ.data .= (im .* g.k) .* op.v̂.data
        transform!(op.w, op.ŵ)
        op.w.data .*= op.U.data
    else
        op.w.data .= op.U.data .* v.data
    end

    transform!(op.ŵ, op.w)

    # ---- linear part and advection ----
    if op.adjoint
        op.v̂.data .= g.λ .* op.v̂.data .+ op.ŵ.data
    else
        op.v̂.data .= g.λ .* op.v̂.data .- (im .* g.k) .* op.ŵ.data
    end

    return transform!(out, op.v̂)
end

linearise!(op::KSLinearised, u::KSField) = (op.U.data .= u.data; op)


# ============================================================================ #
# Preconditioner                                                               #
# ============================================================================ #

# Diagonal preconditioner in Fourier, frozen at the frequency ω₀ and the drift speed c₀ of the initial
# orbit x₀: the field coefficient of the mode (k, n) is divided by one of four multipliers,
#
#     kind = :jacobian    i(ω₀ n - c₀ k) + k⁴ - k², the linear space-time operator about the mean
#                         state u = 0, with 1 on the mean, where it vanishes; the projection
#                         demean! removes that mode
#     kind = :linear      1 + |i(ω₀ n - c₀ k) + k⁴ - k²|, the size of the same operator
#     kind = :frequency   1 + |ω₀ n - c₀ k|, the size of the time derivative in the moving frame
#     kind = :viscous     1 + k⁴, the size of the fourth-order dissipation
#
# The parameters are divided by ‖ω₀ ∂s u₀‖ and ‖∂x u₀‖, their Gauss–Newton curvatures.
struct KSPreconditioner{G<:KSGrid}
    β::Matrix{ComplexF64}    # field multipliers, on the Fourier coefficients
    s::Vector{Float64}       # parameter scales
    û::KSTransformedField{G} # scratch

    function KSPreconditioner(x₀::Orbit; kind::Symbol=:linear)

        # ---- input checks ----
        kind in (:jacobian, :linear, :frequency, :viscous) ||
            throw(ArgumentError("kind must be :jacobian, :linear, :frequency or :viscous"))

        u₀ = x₀.a
        g  = u₀.g
        ω₀ = ReSolver.frequency(x₀)
        c₀ = length(x₀.p) > 1 ? x₀.p[2] : 0.0
        n  = transpose(g.n)

        # ---- field ----
        λ = @. im * (ω₀ * n - c₀ * g.k) + g.k^4 - g.k^2

        β = kind === :jacobian  ? λ :
            kind === :linear    ? @.(1 + abs(λ) + 0im) :
            kind === :frequency ? @.(1 + abs(ω₀ * n - c₀ * g.k) + 0im) :
                                  @.(1 + g.k^4 + 0 * n + 0im)

        # λ₀₀ = 0 on the mean: any nonzero divisor will do, since demean! removes that mode
        kind === :jacobian && (β[1, 1] = 1)

        # ---- parameters ----
        ∂u = similar(u₀)
        s  = [norm(ω₀ .* dds!(∂u, u₀))]
        length(x₀.p) > 1 && push!(s, norm(ddx!(∂u, u₀)))

        return new{typeof(g)}(β, s, KSTransformedField(g))
    end
end

# B is diagonal in Fourier, an orthogonal basis: B⁻⁺ divides by the complex conjugate multipliers,
# and B⁻⁺ = B⁻¹ when these are real
function precondition!(out::Orbit, B::KSPreconditioner, p::Orbit)
    transform!(B.û, p.a)
    B.û.data ./= B.β
    transform!(out.a, B.û)

    out.p .= p.p ./ B.s

    return out
end

function precondition_adjoint!(out::Orbit, B::KSPreconditioner, p::Orbit)
    transform!(B.û, p.a)
    B.û.data ./= conj.(B.β)
    transform!(out.a, B.û)

    out.p .= p.p ./ B.s

    return out
end


# ============================================================================ #
# Projections                                                                  #
# ============================================================================ #

# Projection on the fields of zero space-time mean, P u = u - ⟨u⟩, to be passed as `project` to a
# System. KS is Galilean invariant, u(x, t) → u(x - Ct, t) + C: adding a constant to an orbit and to
# the drift speed gives another solution, a null direction of the Newton system that the phase
# conditions do not remove. The right-hand side is a sum of x-derivatives, so its mean is zero for
# every u, and the equation of the mean, the mode k = n = 0, is void. Both defects lie along the
# constant field, and removing the mean removes both. The inner product is the mean over the grid
# points, so subtracting the mean over the grid points is an orthogonal projection.
function demean!(u::KSField)
    u.data .-= sum(u.data) / length(u.data)

    return u
end

# Projection on the odd fields, u(-x, s) = -u(x, s): the fields fixed by the reflection
# u(x) → -u(-x), under which the KS equation is equivariant, (P u)(x) = (u(x) - u(-x))/2. On the
# grid x_j = (j - 1) L/Nx the reflection maps the point j to j′ = Nx - j + 2, modulo Nx, and fixes
# x = 0, where P u vanishes. P is an orthogonal projection in the inner product of the fields. Odd
# fields have zero mean, so odd! removes the mean too and replaces demean!.
function odd!(u::KSField)
    Nx = u.g.Nx

    for l in axes(u.data, 2), j in 1:(Nx + 1) ÷ 2
        j′ = mod(Nx - j + 1, Nx) + 1
        a  = u.data[j, l]
        b  = u.data[j′, l]

        u.data[j, l]  = (a - b) / 2
        u.data[j′, l] = (b - a) / 2
    end

    return u
end


# ============================================================================ #
# Resolution                                                                   #
# ============================================================================ #

# the field u on the grid g, by spectral interpolation: zero padding or truncation of its Fourier
# coefficients, the highest common modes kept
function resample(u::KSField, g::KSGrid)
    û = rfft(u.data, (1, 2)) ./ (u.g.Nx * u.g.Ns)
    v̂ = zeros(ComplexF64, length(g.k), g.Ns)

    # ---- common modes: k ≥ 0 along x, signed n along s ----
    nk = min(size(û, 1), size(v̂, 1))
    nn = (min(u.g.Ns, g.Ns) - 1) ÷ 2

    v̂[1:nk, 1:nn + 1]       .= û[1:nk, 1:nn + 1]
    v̂[1:nk, end - nn + 1:end] .= û[1:nk, end - nn + 1:end]

    return KSField(g, irfft(v̂ .* (g.Nx * g.Ns), g.Nx, (1, 2)))
end


# ============================================================================ #
# Time integration and near-recurrences, for initial guesses                   #
# ============================================================================ #

# Integrate a snapshot u₀ (a vector on the points along x) for nsteps steps of size dt with ETDRK4
# (Kassam & Trefethen 2005), returning the snapshots every `every` steps as the columns of a
# matrix, the first being u₀. With odd=true the solution is kept odd, u(-x) = -u(x), by removing at
# every step the cosine part of the Fourier coefficients, which rounding errors create and an
# instability of the odd subspace would amplify.
function integrate(u₀::AbstractVector, g::KSGrid, dt::Real, nsteps::Int; every::Int=1,
                   odd::Bool=false)

    # ---- ETDRK4 coefficients, by contour integrals on 32 points ----
    E  = exp.(dt .* g.λ)
    E2 = exp.(dt .* g.λ ./ 2)
    r  = exp.(im .* π .* ((1:32) .- 0.5) ./ 32)
    LR = dt .* g.λ .+ transpose(r)

    Q  = dt .* real.(sum((exp.(LR ./ 2) .- 1) ./ LR; dims=2) ./ 32)
    f1 = dt .* real.(sum((-4 .- LR .+ exp.(LR) .* (4 .- 3LR .+ LR.^2)) ./ LR.^3; dims=2) ./ 32)
    f2 = dt .* real.(sum((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR.^3; dims=2) ./ 32)
    f3 = dt .* real.(sum((-4 .- 3LR .- LR.^2 .+ exp.(LR) .* (4 .- LR)) ./ LR.^3; dims=2) ./ 32)

    # ---- nonlinear term in Fourier, -½ ∂x(u²) ----
    Nl(v̂) = (-im .* g.k ./ 2) .* rfft(irfft(v̂, g.Nx).^2)

    # ---- steps ----
    v̂   = rfft(u₀)
    out = zeros(g.Nx, nsteps ÷ every + 1)
    out[:, 1] .= u₀

    for i in 1:nsteps
        Nv = Nl(v̂)
        â  = E2 .* v̂ .+ Q .* Nv
        Na = Nl(â)
        b̂  = E2 .* v̂ .+ Q .* Na
        Nb = Nl(b̂)
        ĉ  = E2 .* â .+ Q .* (2 .* Nb .- Nv)
        Nc = Nl(ĉ)
        v̂  = E .* v̂ .+ f1 .* Nv .+ 2 .* f2 .* (Na .+ Nb) .+ f3 .* Nc

        # odd fields have purely imaginary coefficients, a sine series
        odd && (v̂ .= im .* imag.(v̂))

        i % every == 0 && (out[:, i ÷ every + 1] .= irfft(v̂, g.Nx))
    end

    return out
end

# u translated by ℓ along x, u(x + ℓ), by a phase shift of its Fourier modes
shift(u::AbstractVector, ℓ::Real, g::KSGrid) = irfft(rfft(u) .* exp.(im .* g.k .* ℓ), g.Nx)

# Best near-recurrence of the snapshots U, spaced by Δt: the start i, the lag m and the shift ℓ
# minimising ‖U[:, i+m] - U(x + ℓ)[:, i]‖ relative to ‖U[:, i]‖, for lags whose duration lies in
# Trange. With drift=false the shift is zero, for periodic orbits. Starts where the trajectory is
# nearly stationary, close to an equilibrium or a travelling wave, are skipped: there the snapshot
# changes by less than a fraction `motion` over a quarter of the shortest lag, up to a shift.
function recurrence(U::AbstractMatrix, g::KSGrid, Δt::Real, Trange; drift::Bool=true,
                    motion::Real=0.3)
    ℓs   = drift ? range(0, g.L; length=129)[1:end-1] : (0.0,)
    lags = ceil(Int, first(Trange) / Δt):floor(Int, last(Trange) / Δt)
    q    = first(lags) ÷ 4

    # change of the snapshot over q steps, minimised over the shifts
    moving(i) = minimum(norm(U[:, i + q] .- shift(U[:, i], ℓ, g)) for ℓ in ℓs) / norm(U[:, i]) > motion

    best = (Inf, 0, 0, 0.0)
    for i in filter(moving, 1:size(U, 2) - last(lags)), ℓ in ℓs
        u = shift(U[:, i], ℓ, g)
        for m in lags
            e = norm(U[:, i + m] .- u) / norm(u)
            e < best[1] && (best = (e, i, m, ℓ))
        end
    end

    return best
end

# Initial orbit from the recurrence (i, m, ℓ) of a trajectory: the segment from U[:, i], over the
# period T = m Δt, resampled on Ns rescaled times by integrating again with steps of T/(Ns × sub),
# and written in the frame drifting with c = -ℓ/T. Returns the Orbit with parameters [log ω, c],
# or [log ω] without drift.
function initial_orbit(U::AbstractMatrix, g::KSGrid, Δt::Real, i::Int, m::Int, ℓ::Real;
                       drift::Bool=true, sub::Int=8)
    T = m * Δt
    c = drift ? -ℓ / T : 0.0

    # ---- the segment on Ns equispaced rescaled times ----
    seg = integrate(U[:, i], g, T / (g.Ns * sub), g.Ns * sub; every=sub)[:, 1:g.Ns]

    # ---- moving frame: a(x, s_l) = u(x + c t_l, t_l) ----
    t = (0:g.Ns - 1) .* (T / g.Ns)
    a = KSField(g, reduce(hcat, [shift(seg[:, l], c * t[l], g) for l in 1:g.Ns]))

    return drift ? Orbit(a, [log(2π / T), c]) : Orbit(a, [log(2π / T)])
end
