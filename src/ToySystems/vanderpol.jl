# Dynamical system for the Van der Pol oscillator, which has a stable limit
# cycle.

# ~~~ state definition ~~~
struct VDPState{N} <: AbstractMatrix{ComplexF64}
    x::Vector{ComplexF64}
    v::Vector{ComplexF64}

    function VDPState{N}(x::Vector{ComplexF64}, v::Vector{ComplexF64}) where {N}
        length(x) == length(v) || throw(ArgumentError("State vectors must be the same length"))
        new{N}(x, v)
    end
end
VDPState(n::Int) = VDPState{n}(Vector{ComplexF64}(undef, (n >> 1) + 1), Vector{ComplexF64}(undef, (n >> 1) + 1))


# ~~~ transforms ~~~
function FFT!(x::VDPState{N}, a::Matrix{Float64}, plan, tmp::Matrix{ComplexF64}) where {N}
    FFTW.unsafe_execute!(plan, a, tmp); tmp .*= 1/N
    x.x .= @view(tmp[:, 1])
    x.v .= @view(tmp[:, 2])
    return x
end

function IFFT!(a::Matrix{Float64}, x::VDPState, plan, tmp::Matrix{ComplexF64})
    tmp[:, 1] .= x.x
    tmp[:, 2] .= x.v
    FFTW.unsafe_execute!(plan, tmp, a)
    return a
end

function FFT(a::Matrix{Float64})
    tmp = rfft(a, [1])
    tmp .*= 1/size(a, 1)
    return VDPState{size(a, 1)}(tmp[:, 1], tmp[:, 2])
end

function IFFT(x::VDPState{N}) where {N}
    tmp = hcat(x.x, x.v)
    return brfft(tmp, N, [1])
end


# ~~~ array interface ~~~
Base.IndexStyle(::Type{<:VDPState})                 = Base.IndexLinear()
Base.eltype(::VDPState)                             = ComplexF64
Base.size(x::VDPState{N}) where {N}                 = ((N >> 1) + 1, 2)
Base.similar(::VDPState{N}, ::Type{T}) where {N, T} = VDPState(N)

function Base.getindex(x::VDPState{N}, i::Int) where {N}
    M = (N >> 1) + 1
    if 0 < i <= M
        return x.x[i]
    elseif i <= 2*M
        return x.v[i - M]
    else
        throw(BoundsError(x, i))
    end
end

function Base.setindex!(x::VDPState{N}, val, i::Int) where {N}
    M = (N >> 1) + 1
    if 0 < i <= M
        x.x[i] = val
    elseif i <= 2*M
        x.v[i - M] = val
    else
        throw(BoundsError(x, i))
    end
end


# ~~~ broadcasting ~~~
const VDPStateStyle = Base.Broadcast.ArrayStyle{VDPState}
Base.BroadcastStyle(::Type{<:VDPState}) = VDPStateStyle()

Base.similar(bc::Base.Broadcast.Broadcasted{VDPStateStyle}, ::Type{T}) where {T} = similar(find_state(bc), T)

find_state(bc::Base.Broadcast.Broadcasted) = find_state(bc.args)
find_state(args::Tuple)                    = find_state(find_state(args[1]), Base.tail(args))
find_state(x)                              = x
find_state(::Tuple{})                      = nothing
find_state(x::VDPState, rest)              = x
find_state(::Any, rest)                    = find_state(rest)


# ~~~ linear algebra ~~~
function LinearAlgebra.dot(x::VDPState{N}, y::VDPState{N}) where {N}
    # output
    out = 0.0

    # loop over frequencies adding to sum
    for n in 2:((N >> 1) + 1)
        out += real(dot(x.x[n], y.x[n]))
        out += real(dot(x.v[n], y.v[n]))
    end

    # add mean component
    out += 0.5*real(dot(x.x[1], y.x[1]))
    out += 0.5*real(dot(x.v[1], y.v[1]))

    return out
end
LinearAlgebra.norm(x::VDPState) = sqrt(dot(x, x))


# ~~~ operators ~~~
# compute tangent state vector
function dds!(dxds::V, x::V) where {N, V<:VDPState{N}}
    for n in 1:((N >> 1) + 1)
        dxds.x[n] = 1im*(n - 1)*x.x[n]
        dxds.v[n] = 1im*(n - 1)*x.v[n]
    end
    return dxds
end

struct VDPSystem{FP, IFP}
    mu::Float64
    tmp1::Matrix{Float64}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}
    tmp4::Matrix{ComplexF64}
    fplan::FP
    iplan::IFP

    function VDPSystem(n::Int, mu)
        fplan = plan_rfft(Matrix{Float64}(undef, n, 2), [1], flags=FFTW.ESTIMATE)
        iplan = plan_brfft(Matrix{ComplexF64}(undef, (n >> 1) + 1, 2), n, [1], flags=FFTW.ESTIMATE)
        new{typeof(fplan), typeof(iplan)}(mu,
                                         [zeros(Float64, n, 2) for _ in 1:3]...,
                                          zeros(ComplexF64, (n >> 1) + 1, 2),
                                          fplan,
                                          iplan)
    end
end
VDPSystem(::VDPState{N}, mu) where {N} = VDPSystem(N, mu)

function (sys::VDPSystem)(F::V, x::V) where {V<:VDPState}
    # aliases
    μ = sys.mu
    a = sys.tmp1
    f = sys.tmp2

    # transform input
    IFFT!(a, x, sys.iplan, sys.tmp4)

    # compute response
    @views f[:, 1] .= a[:, 2]
    @views f[:, 2] .= μ.*(1 .- a[:, 1].^2).*a[:, 2] .- a[:, 1]

    return FFT!(F, f, sys.fplan, sys.tmp4)
end

function (sys::VDPSystem)(G::V, x::V, r::V) where {V<:VDPState}
    # aliases
    μ = sys.mu
    a = sys.tmp1
    s = sys.tmp2
    g = sys.tmp3

    # transform input
    IFFT!(a, x, sys.iplan, sys.tmp4)
    IFFT!(s, r, sys.iplan, sys.tmp4)

    # compute response
    @views g[:, 1] .= .-(2.0.*μ.*a[:, 1].*a[:, 2] .+ 1).*s[:, 2]
    @views g[:, 2] .= s[:, 1] .+ μ.*(1 .- a[:, 1].^2).*s[:, 2]

    return FFT!(G, g, sys.fplan, sys.tmp4)
end
