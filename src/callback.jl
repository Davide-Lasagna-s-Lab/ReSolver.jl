# This file contains the definitions to create a default callback method
# extending the default behaviour of Optim.jl.

struct Callback{T, O}
    trace::T
    opts::O
    start_itr::Int
    start_time::Float64

    function Callback(trace::T, opts::O) where {T, O}
        start_itr = _get_start_itr(trace)
        start_time = _get_start_time(trace)
        new{T, O}(trace, opts, start_itr, start_time)
    end
end

function (f::Callback)(state)
    # unpack current state
    curr_itr = state.iteration
    curr_time = state.metadata["time"]
    curr_step = state.metadata["Current step size"]
    curr_residual = state.value
    curr_g_norm = state.g_norm
    curr_x = state.metadata["x"]

    # update trace
    push!(f.trace, curr_x, curr_itr, curr_time, curr_step, curr_residual, curr_g_norm, f.start_itr, f.start_time)

    # print state
    if f.opts.verbose && itr % opts.n_it_print == 0
        print_state(f.opts.io, f.trace[end])
    end

    # check for convergence and call user defined callback
    return curr_residual < f.opts.res_tol || f.opts.callback(curr_x, f.trace[end])
end
