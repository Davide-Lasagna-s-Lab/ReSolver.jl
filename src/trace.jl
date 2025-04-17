# Interface to keep track of optimisation variables during the course of an
# optimisation.

struct OptimTrace{S, F}
    itrs::Vector{Int}
    times::Vector{Float64}
    steps::Vector{Float64}
    residuals::Vector{Float64}
    g_norms::Vector{Float64}
    others::S
    f::F

    function OptimTrace(x, f::F) where {F}
        # initialise main trace vectors
        itrs      = Int[]
        times     = Float64[]
        steps     = Float64[]
        residuals = Float64[]
        g_norms   = Float64[]

        # check output of provided function
        other = f(x)
        others = isnothing(other) ? nothing : typeof(other)[]

        new{typeof(other), F}(itrs, times, steps, residuals, g_norms, others, f)
    end
end

OptimTrace(x) = OptimTrace(x, x->nothing)


Base.IndexStyle(::Type{<:OptimTrace}) = IndexLinear()
Base.length(t::OptimTrace) = length(t.itrs)

@inline function Base.getindex(t::OptimTrace, i::Int)
    @boundscheck checkbounds(t.itrs, i)
    @inbounds state = (t.itrs[i], t.times[i], t.steps[i], t.residuals[i], t.g_norms[i], t.others[i])
    return state
end

@inline itrs(t::OptimTrace)      = t.itrs
@inline times(t::OptimTrace)     = t.times
@inline steps(t::OptimTrace)     = t.steps
@inline residuals(t::OptimTrace) = t.residuals
@inline g_norms(t::OptimTrace)   = t.gradients
@inline others(t::OptimTrace)    = t.others


function Base.push!(t::OptimTrace, curr_x, curr_itr, curr_time, curr_step, curr_residual, curr_g_norm, start_itr, start_time)
    # push the values to main trace vectors
    push!(t.itrs, curr_itr + start_itr)
    push!(t.times, curr_time + start_time)
    push!(t.steps, curr_step)
    push!(t.residuals, curr_residual)
    push!(t.g_norms, curr_g_norm)

    # push value to the other trace vector
    _update_other!(t.others, t.f(curr_x))

    return t
end
_update_other!(others::Vector{T}, curr_other::T) where {T} = push!(others, curr_other)
_update_other!(::Nothing, curr_other)                      = nothing


function print_state(io, state::Tuple{Int, Float64, Float64, Float64, Float64, Float64, T}) where {T}
    @printf io "|%10d   |  %5.5e  |   %5.2e  |  %5.5e  |  %5.5e  |" state[1] state[2] state[3] state[4] state[4]
    flush(io)
end

function print_header(io, trace)
    println(io, "-----------------------------------------------------------------------------")
    println(io, "|  Iteration  |     Time      |  Step Size  |   Residual    |   Gradient    |")
    println(io, "-----------------------------------------------------------------------------")
    flush(io)
end


function _get_start_time(trace)
    try
        return times(trace)[end]
    catch
        return 0.0
    end
end

function _get_start_itr(trace)
    try
        return iters(trace)[end]
    catch
        return 0 
    end
end

function _get_final_itr(trace)
    try
        return itrs(trace)[end]
    catch
        return -1
    end
end
