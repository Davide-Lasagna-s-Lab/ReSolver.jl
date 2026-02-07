# This file contains the definitions to create a default callback method
# extending the default behaviour of Optim.jl.

struct CallbackCache{S, O}
         state::S
          opts::O
     start_itr::Int
    start_time::Float64

    function CallbackCache(state::S, opts::O) where {S, O}
        new{S, O}(state, opts, state.itr, state.time)
    end
end

function (f::CallbackCache)(state)
    # unpack current state
    curr_itr      = state.iteration
    curr_time     = state.metadata["time"]
    curr_step     = state.metadata["Current step size"]
    curr_residual = state.value
    curr_g_norm   = state.g_norm
    curr_optvec   = state.metadata["x"]

    # unpack current optimisation vector
    curr_x = curr_optvec.x
    curr_T = curr_optvec.T

    # update trace
    update!(f.state, curr_x,
                     curr_T,
                     curr_itr,
                     curr_time,
                     curr_step,
                     curr_residual,
                     curr_g_norm,
                     f.start_itr,
                     f.start_time)

    # print state
    if f.opts.verbose && curr_itr % f.opts.n_it_print == 0
        print_state(f.opts.io, f.state)
    end

    # check for convergence and call user defined callback
    return curr_residual < f.opts.res_tol || f.opts.callback(f.state)
end
