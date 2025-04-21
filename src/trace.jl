# Interface to keep track of optimisation variables during the course of an
# optimisation.

export OptTrace,
       itrs,
       times,
       steps,
       residuals,
       g_norms,
       periods,
       others

# ~~~ Optimisation trace ~~~
struct OptTrace{S, F}
         itrs::Vector{Int}
        times::Vector{Float64}
        steps::Vector{Float64}
    residuals::Vector{Float64}
      g_norms::Vector{Float64}
      periods::Vector{Float64}
       others::Vector{S}
            f::F

    function OptTrace(x, f::F) where {F}
        # initialise main trace vectors
        itrs      = Int[]
        times     = Float64[]
        steps     = Float64[]
        residuals = Float64[]
        g_norms   = Float64[]
        periods   = Float64[]

        # check output of provided function
        other = f(x)
        others = isnothing(other) ? Nothing[] : typeof(other)[]

        new{typeof(other), F}(itrs, times, steps, residuals, g_norms, periods, others, f)
    end
end
OptTrace(x) = OptTrace(x, x->nothing)


# ~~~ Trace interface ~~~
Base.IndexStyle(::Type{<:OptTrace}) = IndexLinear()
Base.length(t::OptTrace) = length(t.itrs)
Base.lastindex(t::OptTrace) = length(t)

@inline function Base.getindex(t::OptTrace, i::Int)
    @boundscheck checkbounds(t.itrs, i)
    @inbounds state = (t.itrs[i], t.times[i], t.steps[i], t.residuals[i], t.g_norms[i], t.periods[i], _get_other(t.others, i))
    return state
end
_get_other(others, i)            = others[i]
_get_other(::Vector{Nothing}, i) = nothing

@inline itrs(t::OptTrace)      = t.itrs
@inline times(t::OptTrace)     = t.times
@inline steps(t::OptTrace)     = t.steps
@inline residuals(t::OptTrace) = t.residuals
@inline g_norms(t::OptTrace)   = t.g_norms
@inline periods(t::OptTrace)   = t.periods
@inline others(t::OptTrace)    = t.others

function Base.push!(t::OptTrace, curr_x, curr_itr, curr_time, curr_step, curr_residual, curr_g_norm, curr_period, start_itr, start_time)
    # push the values to main trace vectors
    push!(t.itrs,      curr_itr + start_itr)
    push!(t.times,     curr_time + start_time)
    push!(t.steps,     curr_step)
    push!(t.residuals, curr_residual)
    push!(t.g_norms,   curr_g_norm)
    push!(t.periods,   curr_period)

    # push value to the other trace vector
    _update_other!(t.others, t.f, curr_x)

    return t
end
_update_other!(others, f, x)            = push!(others, f(x))
_update_other!(::Vector{Nothing}, f, x) = nothing


# ~~~ State printing ~~~
function print_state(io, state::Tuple{Int, Float64, Float64, Float64, Float64, Float64, T}) where {T}
    @printf io "|%10d   |  %5.5e  |   %5.2e  |  %5.5e  |  %5.5e  |    %8.4f   |\n" state[1] state[2] state[3] state[4] state[5] state[6]
    flush(io)
end

function print_header(io, trace)
    println(io, "---------------------------------------------------------------------------------------------")
    println(io, "|  Iteration  |     Time      |  Step Size  |   Residual    |   Gradient    |     Period    |")
    println(io, "---------------------------------------------------------------------------------------------")
    flush(io)
end


# ~~~ Utility functions ~~~
function _get_start_time(trace)
    try
        return times(trace)[end]
    catch
        return 0.0
    end
end

function _get_start_itr(trace)
    try
        return itrs(trace)[end]
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
