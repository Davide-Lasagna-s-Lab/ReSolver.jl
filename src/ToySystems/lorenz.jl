# Dynamical system for the Lorenz system, which has a stable limit, which
# exhibits chaotic dynamics.

# ~~~ state definition ~~~
struct LorenzState{N} <: AbstractMatrix{ComplexF64}
    x::Vector{ComplexF64}
    y::Vector{ComplexF64}
    z::Vector{ComplexF64}

    function LorenzState{N}(x::Vector{ComplexF64}, y::Vector{ComplexF64}, z::Vector{ComplexF64}) where {N}
        length(x) == length(y) == length(z) || throw(ArgumentError("State vectors must be the same length"))
        new{N}(x, y, z)
    end
end
LorenzState(n::Int) = LorenzState{n}([Vector{ComplexF64}(undef, (n >> 1) + 1) for _ in 1:3]...)


# ~~~ transforms ~~~
function FFT!(u::LorenzState{N}, a::Matrix{Float64}, plan, tmp::Matrix{ComplexF64}) where {N}
    FFTW.unsafe_execute!(plan, a, tmp); tmp .*= 1/N
    u.x .= @view(tmp[:, 1])
    u.y .= @view(tmp[:, 2])
    u.z .= @view(tmp[:, 3])
    return u
end

function IFFT!(a::Matrix{Float64}, u::LorenzState, plan, tmp::Matrix{ComplexF64})
    tmp[:, 1] .= u.x
    tmp[:, 2] .= u.y
    tmp[:, 3] .= u.z
    FFTW.unsafe_execute!(plan, tmp, a)
    return a
end

function FFT2(a::Matrix{Float64})
    tmp = rfft(a, [1])
    tmp .*= 1/size(a, 1)
    return LorenzState{size(a, 1)}(tmp[:, 1], tmp[:, 2], tmp[:, 3])
end

function IFFT(u::LorenzState{N}, pad::Int=0) where {N}
    tmp = hcat(vcat(u.x, zeros(pad)),
               vcat(u.y, zeros(pad)),
               vcat(u.z, zeros(pad)))
    return brfft(tmp, (size(tmp, 1) - 1) << 1, [1])
end


# ~~~ array interface ~~~
Base.IndexStyle(::Type{<:LorenzState})                 = Base.IndexLinear()
Base.eltype(::LorenzState)                             = ComplexF64
Base.size(::LorenzState{N}) where {N}                  = ((N >> 1) + 1, 3)
Base.similar(::LorenzState{N}, ::Type{T}) where {N, T} = LorenzState(N)

function Base.getindex(u::LorenzState{N}, i::Int) where {N}
    M = (N >> 1) + 1
    if 0 < i <= M
        return u.x[i]
    elseif i <= 2*M
        return u.y[i - M]
    elseif i <= 3*M
        return u.z[i - 2*M]
    else
        throw(BoundsError(u, i))
    end
end

function Base.setindex!(u::LorenzState{N}, val, i::Int) where {N}
    M = (N >> 1) + 1
    if 0 < i <= M
        u.x[i] = val
    elseif i <= 2*M
        u.y[i - M] = val
    elseif i <= 3*M
        u.z[i - 2*M] = val
    else
        throw(BoundsError(u, i))
    end
end


# ~~~ broadcasting ~~~
const LorenzStateStyle = Base.Broadcast.ArrayStyle{LorenzState}
Base.BroadcastStyle(::Type{<:LorenzState}) = LorenzStateStyle()

Base.similar(bc::Base.Broadcast.Broadcasted{LorenzStateStyle}, ::Type{T}) where {T} = similar(find_state(bc), T)

# the rest of the broadcasting interface is available in vanderpol.jl
find_state(u::LorenzState, rest) = u


# ~~~ linear algebra ~~~
function LinearAlgebra.dot(u::LorenzState{N}, v::LorenzState{N}) where {N}
    # output
    out = 0.0

    # loop over frequencies adding to sum
    for n in 2:((N >> 1) + 1)
        out += 2*real(dot(u.x[n], v.x[n]))
        out += 2*real(dot(u.y[n], v.y[n]))
        out += 2*real(dot(u.z[n], v.z[n]))
    end

    # add mean component
    out += real(dot(u.x[1], v.x[1]))
    out += real(dot(u.y[1], v.y[1]))
    out += real(dot(u.z[1], v.z[1]))

    return out
end
LinearAlgebra.norm(u::LorenzState) = sqrt(dot(u, u))


# ~~~ operators ~~~
# compute tangent state vector
function dds!(duds::V, u::V) where {N, V<:LorenzState{N}}
    for n in 1:((N >> 1) + 1)
        duds.x[n] = 1im*(n - 1)*u.x[n]
        duds.y[n] = 1im*(n - 1)*u.y[n]
        duds.z[n] = 1im*(n - 1)*u.z[n]
    end
    return duds
end

struct LorenzSystem{FP, IFP}
    rho::Float64
    sigma::Float64
    beta::Float64
    tmp1::Matrix{Float64}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}
    tmp4::Matrix{ComplexF64}
    fplan::FP
    iplan::IFP

    function LorenzSystem(n::Int; rho::Float64=28.0, sigma::Float64=10.0, beta::Float64=8/3)
        fplan = plan_rfft(Matrix{Float64}(undef, n, 3), [1], flags=FFTW.ESTIMATE)
        iplan = plan_brfft(Matrix{ComplexF64}(undef, (n >> 1) + 1, 3), n, [1], flags=FFTW.ESTIMATE)
        new{typeof(fplan), typeof(iplan)}(rho, sigma, beta,
                                         [zeros(Float64, n, 3) for _ in 1:3]...,
                                          zeros(ComplexF64, (n >> 1) + 1, 3),
                                          fplan,
                                          iplan)
    end
end
LorenzSystem(::LorenzState{N}; kwargs...) where {N} = LorenzSystem(N, kwargs...)

function (sys::LorenzSystem)(F::V, u::V) where {V<:LorenzState}
    # aliases
    ρ = sys.rho
    σ = sys.sigma
    β = sys.beta
    a = sys.tmp1
    f = sys.tmp2

    # transform input
    IFFT!(a, u, sys.iplan, sys.tmp4)

    # compute response
    @views f[:, 1] .= σ.*(a[:, 2] .- a[:, 1])
    @views f[:, 2] .= a[:, 1].*(ρ .- a[:, 3]) .- a[:, 2]
    @views f[:, 3] .= a[:, 1].*a[:, 2] .- β.*a[:, 3]

    return FFT!(F, f, sys.fplan, sys.tmp4)
end

function (sys::LorenzSystem)(G::V, u::V, r::V) where {V<:LorenzState}
    # aliases
    ρ = sys.rho
    σ = sys.sigma
    β = sys.beta
    a = sys.tmp1
    s = sys.tmp2
    g = sys.tmp3

    # transform input
    IFFT!(a, u, sys.iplan, sys.tmp4)
    IFFT!(s, r, sys.iplan, sys.tmp4)

    # compute response
    @views g[:, 1] .= -σ.*s[:, 1] .+ (ρ .- a[:, 3]).*s[:, 2] .+ a[:, 2].*s[:, 3]
    @views g[:, 2] .=  σ.*s[:, 1] .-                 s[:, 2] .+ a[:, 1].*s[:, 3]
    @views g[:, 3] .=             .-        a[:, 1].*s[:, 2] .-       β.*s[:, 3]

    return FFT!(G, g, sys.fplan, sys.tmp4)
end
