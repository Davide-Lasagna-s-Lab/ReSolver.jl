# Optimisation options

@with_kw struct OptOptions{OPTIMIZER<:AbstractOptimizer, CB}
    # general options
    maxiter::Int        = typemax(Int)                                  # maximum number of iterations
    alg::OPTIMIZER      = LBFGS()                                       # optimisation algorithm choice
    res_tol::Float64    = 1e-12                                         # residual tolerance used to determine if the solution is converged
    time_limit::Float64 = NaN                                           # time limit on the optimisation (in seconds)
    callback::CB        = x->false; @assert !isempty(methods(callback)) # user specified callback function to be executed every iteration

    # optim.jl options
    x_atol::Float64           = 0.0                                     # optimisation variable absolute tolerance
    x_rtol::Float64           = 0.0                                     # optimisation variable relative tolerance
    f_atol::Float64           = 0.0                                     # objective function tolerance
    f_rtol::Float64           = 0.0                                     # objective function tolerance
    g_atol::Float64           = 0.0                                     # gradient tolerance
    f_calls_limit::Int        = 0                                       # limit on the number of calls to the objective function
    g_calls_limit::Int        = 0                                       # limit on the number of calls to the gradient function
    allow_f_increases::Bool   = false                                   # allow the objective to increase between iterations
    successive_f_tol::Int     = 1                                       # number of times the objective is allowed to increase accross iterations
    show_optim_warnings::Bool = false                                   # optionally show warnings from Optim.jl options

    # printing options
    verbose::Bool   = true                                              # whether to print the state of the optimisation at each iteration
    n_it_print::Int = 1                                                 # number of iterations between printing the state of the optimisation
    io::IO          = stdout                                            # IO stream to print the state to
end

# convert to options for Optim.jl
genOptimOptions(opts, state) = Options(x_abstol          = opts.x_atol,
                                       x_reltol          = opts.x_rtol,
                                       f_abstol          = opts.f_atol,
                                       f_reltol          = opts.f_rtol,
                                       g_abstol          = opts.g_atol,
                                       f_calls_limit     = opts.f_calls_limit,
                                       g_calls_limit     = opts.g_calls_limit,
                                       allow_f_increases = opts.allow_f_increases,
                                       successive_f_tol  = opts.successive_f_tol,
                                       iterations        = opts.maxiter,
                                       time_limit        = opts.time_limit,
                                       callback          = CallbackCache(state, opts),
                                       store_trace       = false,
                                       trace_simplex     = false,
                                       show_trace        = false,
                                       extended_trace    = true,
                                       show_every        = 1,
                                       show_warnings     = opts.show_optim_warnings)
