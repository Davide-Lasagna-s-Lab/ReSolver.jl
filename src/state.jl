# Interface to keep track of optimisation variables during the course of an
# optimisation.

export OptState,
       itr,
       time,
       step,
       residual,
       g_norm,
       period,
       other

# ~~~ Optimisation trace ~~~
mutable struct OptState{X, TX, S, F}
     const x::X
      period::TX
         itr::Int
        time::Float64
        step::Float64
    residual::TX
      g_norm::TX
       other::S
           f::F

    function OptState(x::X, f::F) where {X, F}
        # initialise state variables
        itr      = 0
        time     = 0.0
        step     = 0.0
        residual = real(eltype(x))(Inf)
        g_norm   = real(eltype(x))(Inf)
        period   = real(eltype(x))(Inf)
        other    = f(x)

        new{X, real(eltype(x)), typeof(other), F}(x, period, itr, time, step, residual, g_norm, other, f)
    end
end
OptState(x) = OptState(x, x->nothing)

@inline        x(s::OptState) = s.x
@inline   period(s::OptState) = s.period
@inline      itr(s::OptState) = s.itr
@inline     time(s::OptState) = s.time
@inline     step(s::OptState) = s.step
@inline residual(s::OptState) = s.residual
@inline   g_norm(s::OptState) = s.g_norm
@inline    other(s::OptState) = s.other

function update!(s::OptState{X, TX}, curr_x::X,
                                     curr_period::TX,
                                     curr_itr,
                                     curr_time,
                                     curr_step,
                                     curr_residual::TX,
                                     curr_g_norm::TX,
                                     start_itr::Int,
                                     start_time) where {X, TX}
    # push the values to main trace vectors
    s.x       .= curr_x
    s.period   = curr_period
    s.itr      = curr_itr + start_itr
    s.time     = curr_time + start_time
    s.step     = curr_step
    s.residual = curr_residual
    s.g_norm   = curr_g_norm
    s.other    = s.f(curr_x)

    return s
end


# ~~~ State printing ~~~
function print_state(io, state)
    @printf io "|%10d   |  %5.5e  |   %5.2e  |  %5.5e  |  %5.5e  |    %8.4f   |\n" state.itr state.time state.step state.residual state.g_norm state.period
    flush(io)
end

function print_header(io)
    println(io, "---------------------------------------------------------------------------------------------")
    println(io, "|  Iteration  |     Time      |  Step Size  |   Residual    |   Gradient    |     Period    |")
    println(io, "---------------------------------------------------------------------------------------------")
    flush(io)
end
