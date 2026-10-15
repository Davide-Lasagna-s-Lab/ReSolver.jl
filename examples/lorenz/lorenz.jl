# The Lorenz system in space-time form, for ReSolver.
#
#     ẋ = σ (y - x),   ẏ = x (ρ - z) - y,   ż = x y - β z,     σ = 10, ρ = 28, β = 8/3.
#
# A space-time field is a LorenzField: the Fourier coefficients û_n, n = 0, …, K, of the state
# (x, y, z) in rescaled time s, with u(s) = û_0 + 2 Re Σ_{n≥1} û_n e^{ins}. The time derivative
# and the preconditioner act mode by mode on them; the nonlinear terms are computed on a grid of
# 4K + 1 points in s, which removes the aliasing of the quadratic products. The inner product is
# the mean over s of the products of the states, û_0 v̂_0 + 2 Re Σ_{n≥1} conj(û_n) v̂_n on the
# coefficients: the derivative is skew-adjoint and the adjoint operator exact in it.
#
# The file provides what a ReSolver System needs, and the pieces to build an initial guess:
#
#     LorenzField(K)                    space-time field with K modes, zero
#     nonlinear!, LorenzLinearised      N(u), its linearisation and adjoint
#     linearise!(op, u)                 linearisation point of LorenzLinearised
#     dds!(out, u)                      derivative in rescaled time s
#     LorenzPreconditioner              block preconditioner, one 3 × 3 block per mode
#     resample(u, K)                    the field with K modes
#     trajectory, initial_orbit         time integration (RK4) and initial guesses

using FFTW
using LinearAlgebra
using ReSolver

import ReSolver: precondition!, precondition_adjoint!

const σ = 10.0
const ρ = 28.0
const β = 8 / 3


# ============================================================================ #
# Fields                                                                       #
# ============================================================================ #

"""
    LorenzField(K)
    LorenzField(û)

Fourier coefficients in rescaled time of a periodic state of the Lorenz system: a 3 × (K + 1)
complex matrix, one row per variable, one column per mode n = 0, …, K.
"""
struct LorenzField <: AbstractMatrix{ComplexF64}
    data::Matrix{ComplexF64} # coefficients û_n, 3 × (K + 1)

    LorenzField(û::Matrix{ComplexF64}) = new(û)
    LorenzField(K::Int)                = new(zeros(ComplexF64, 3, K + 1))
end

# ---- array interface ----
Base.size(u::LorenzField)                    = size(u.data)
Base.getindex(u::LorenzField, I::Int...)     = u.data[I...]
Base.setindex!(u::LorenzField, v, I::Int...) = (u.data[I...] = v)
Base.IndexStyle(::Type{LorenzField})         = IndexLinear()
Base.similar(u::LorenzField)                 = LorenzField(similar(u.data))
Base.copy(u::LorenzField)                    = LorenzField(copy(u.data))

# number of modes beyond the mean
modes(u::LorenzField) = size(u, 2) - 1

# mean over s of the products of the states, from the coefficients
function LinearAlgebra.dot(u::LorenzField, v::LorenzField)
    w = [1; fill(2, modes(u))]
    return real(sum(conj.(u.data) .* v.data .* transpose(w)))
end

LinearAlgebra.norm(u::LorenzField) = sqrt(dot(u, u))

# ---- broadcasting keeps the field type ----
Base.BroadcastStyle(::Type{LorenzField}) = Broadcast.ArrayStyle{LorenzField}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{LorenzField}}, ::Type{ComplexF64}) =
    LorenzField(similar(Array{ComplexF64}, axes(bc)))

# ---- transforms between the coefficients and the 4K + 1 points of the grid ----
npoints(u::LorenzField) = 4modes(u) + 1

function topoints(u::LorenzField)
    M = npoints(u)
    û = zeros(ComplexF64, 3, M ÷ 2 + 1)
    û[:, 1:modes(u) + 1] .= u.data

    return irfft(û .* M, M, 2)
end

function tocoefficients!(out::LorenzField, v::Matrix{Float64})
    out.data .= (rfft(v, 2) ./ size(v, 2))[:, 1:modes(out) + 1]
    return out
end


# ============================================================================ #
# Operators                                                                    #
# ============================================================================ #

# right-hand side of the Lorenz system, on the grid
function nonlinear!(out::LorenzField, u::LorenzField)
    v = topoints(u)
    x = v[1, :]
    y = v[2, :]
    z = v[3, :]

    f = permutedims([σ .* (y .- x)  x .* (ρ .- z) .- y  x .* y .- β .* z])

    return tocoefficients!(out, f)
end

# Jacobian of the right-hand side about U, or its transpose, applied on the grid
struct LorenzLinearised
          U::Matrix{Float64} # linearisation point on the grid, set by linearise!
    adjoint::Bool            # false: J(U) v, true: J(U)ᵀ w

    LorenzLinearised(K::Int; adjoint::Bool=false) = new(zeros(3, 4K + 1), adjoint)
end

# The Jacobian of the Lorenz system at (x, y, z) is
#
#     J = [ -σ      σ     0
#           ρ - z  -1    -x
#           y       x    -β ],
#
# applied to the components (a, b, c) of w at every point, written out to avoid allocations.
function (op::LorenzLinearised)(out::LorenzField, v::LorenzField)
    w = topoints(v)

    for j in axes(w, 2)
        x, y, z = op.U[1, j], op.U[2, j], op.U[3, j]
        a, b, c = w[1, j], w[2, j], w[3, j]

        if op.adjoint
            # ---- Jᵀ w ----
            w[1, j] = -σ * a + (ρ - z) * b + y * c
            w[2, j] =  σ * a - b + x * c
            w[3, j] = -x * b - β * c
        else
            # ---- J w ----
            w[1, j] = σ * (b - a)
            w[2, j] = (ρ - z) * a - b - x * c
            w[3, j] = y * a + x * b - β * c
        end
    end

    return tocoefficients!(out, w)
end

linearise!(op::LorenzLinearised, u::LorenzField) = (op.U .= topoints(u); op)

# ∂s, diagonal on the coefficients
dds!(out::LorenzField, u::LorenzField) = (out.data .= u.data .* transpose(im .* (0:modes(u))); out)

# Block-diagonal preconditioner, one 3 × 3 block per mode n, of one of two kinds:
#
#     kind = :jacobian    i ω₀ n I - J̄, the linear space-time operator with the Jacobian frozen at
#                         the mean state of the initial orbit
#     kind = :frequency   (1 + ω₀ n) I, the size of the time derivative only
#
# The log-frequency is divided by ‖ω₀ ∂s u₀‖, its Gauss–Newton curvature. The modes are orthogonal
# in the inner product, so the adjoint of B⁻¹ is the conjugate transpose of each block.
struct LorenzPreconditioner
    F::Vector{LU{ComplexF64, Matrix{ComplexF64}, Vector{Int}}} # factorised blocks, n = 0, …, K
    s::Float64                                                 # scale of the log-frequency

    function LorenzPreconditioner(x₀::Orbit; kind::Symbol=:jacobian)

        # ---- input checks ----
        kind in (:jacobian, :frequency) ||
            throw(ArgumentError("kind must be :jacobian or :frequency"))

        ω₀ = ReSolver.frequency(x₀)
        K  = modes(x₀.a)

        # ---- Jacobian at the mean state ----
        x, y, z = real.(x₀.a.data[:, 1])
        J̄ = [ -σ     σ    0
             ρ - z  -1   -x
               y     x   -β ]

        # ---- blocks and scale ----
        block(n) = kind === :jacobian ? im * ω₀ * n * I - J̄ : (1 + ω₀ * n) * Matrix{ComplexF64}(I, 3, 3)

        F = [lu(block(n)) for n in 0:K]
        s = norm(ω₀ .* dds!(similar(x₀.a), x₀.a))

        return new(F, s)
    end
end

function precondition!(out::Orbit, B::LorenzPreconditioner, p::Orbit)
    for (j, F) in enumerate(B.F)
        out.a.data[:, j] .= F \ p.a.data[:, j]
    end
    out.p .= p.p ./ B.s

    return out
end

function precondition_adjoint!(out::Orbit, B::LorenzPreconditioner, p::Orbit)
    for (j, F) in enumerate(B.F)
        out.a.data[:, j] .= F' \ p.a.data[:, j]
    end
    out.p .= p.p ./ B.s

    return out
end


# ============================================================================ #
# Initial guess                                                                #
# ============================================================================ #

# one RK4 step of the Lorenz system
function rk4(u::Vector, dt::Real)
    f(u) = [σ * (u[2] - u[1]), u[1] * (ρ - u[3]) - u[2], u[1] * u[2] - β * u[3]]

    k1 = f(u)
    k2 = f(u .+ dt / 2 .* k1)
    k3 = f(u .+ dt / 2 .* k2)
    k4 = f(u .+ dt .* k3)

    return u .+ dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

# nsteps steps from u, returning every state as a column
function trajectory(u::Vector, dt::Real, nsteps::Int)
    U = zeros(3, nsteps + 1)
    U[:, 1] .= u

    for i in 1:nsteps
        U[:, i + 1] .= rk4(U[:, i], dt)
    end

    return U
end

# Initial orbit with K modes from the near-recurrence (i, m) of a trajectory U with steps dt: the
# segment from U[:, i], over the period T = m dt, sampled on the 4K + 1 points of the grid by
# integrating again, then its coefficients. Returns the Orbit with parameters [log ω].
function initial_orbit(U::AbstractMatrix, dt::Real, i::Int, m::Int, K::Int)
    T   = m * dt
    M   = 4K + 1
    sub = 20
    seg = trajectory(U[:, i], T / (M * sub), M * sub)[:, 1:sub:end][:, 1:M]

    return Orbit(tocoefficients!(LorenzField(K), seg), [log(2π / T)])
end

# the field u with K modes, by zero padding or truncation of its coefficients
function resample(u::LorenzField, K::Int)
    v = LorenzField(K)
    n = min(modes(u), K) + 1
    v.data[:, 1:n] .= u.data[:, 1:n]

    return v
end

# the orbit on np points of rescaled time, closed
function curve(x::Orbit, np::Int=400)
    û = zeros(ComplexF64, 3, np ÷ 2 + 1)
    û[:, 1:modes(x.a) + 1] .= x.a.data
    v = irfft(û .* np, np, 2)

    return [v v[:, 1]]
end
