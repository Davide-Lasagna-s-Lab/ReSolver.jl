# Optimisation options

export OptOptions

@with_kw struct OptOptions{OPTIMIZER<:AbstractOptimizer, CB}
    # general options
    maxiter::Int = typemax(Int)                                      # maximum number of iterations
    alg::OPTIMIZER = LBFGS()                                         # optimisation algorithm choice
    res_tol::Float64 = 1e-12                                         # residual tolerance used to determine if the solution is converged
    time_limit::Float64 = NaN                                        # time limit on the optimisation (in seconds)
    callback::CB = x->false; @assert !isempty(methods(callback))     # user specified callback function to be executed every iteration

    # optim.jl options
    g_tol::Float64 = 0.0                                             # gradient tolerance
    x_tol::Float64 = 0.0                                             # optimisation variable tolerance
    f_tol::Float64 = 0.0                                             # objective function tolerance
    f_calls_limit::Int = 0                                           # limit on the number of calls to the objective function
    g_calls_limit::Int = 0                                           # limit on the number of calls to the gradient function
    allow_f_increases::Bool = false                                  # allow the objective to increase between iterations

    # printing options
    verbose::Bool = true                                             # whether to print the state of the optimisation at each iteration
    n_it_print::Int = 1                                              # number of iterations between printing the state of the optimisation
    io::IO = stdout                                                  # IO stream to print the state to
end

# convert to options for Optim.jl
# TODO: update options for complete set
genOptimOptions(opts, trace) = Options(g_tol=opts.g_tol,
                                       x_abstol=opts.x_tol,
                                       f_abstol=opts.f_tol,
                                       f_calls_limit=opts.f_calls_limit,
                                       g_calls_limit=opts.g_calls_limit,
                                       trace_simplex=false,
                                       allow_f_increases=opts.allow_f_increases,
                                       iterations=opts.maxiter,
                                       show_trace=false,
                                       extended_trace=true,
                                       show_every=1,
                                       time_limit=opts.time_limit,
                                       store_trace=false,
                                       callback=CallbackCache(trace, opts))
