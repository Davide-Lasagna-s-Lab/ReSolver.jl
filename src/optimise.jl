# Interface for the optimisation of a type using the residuals as objective.

# TODO: interface with OptimKit.jl
# TODO: optional residual constructor with no time derivative
# TODO: default finite difference gradient operator

# ~~~ Assumed interface for type X ~~~
#  - eltype(::X) -> Type
#  - similar(::X, ::Type{T}) -> X
#  - dot(::X, ::X) -> Number
#  - norm(::X) -> Non-negative Real
#  - broadcasting between variables of type X
#  - dds!(::X, ::X) -> X
#  - rhs!(::X, ::X) -> X
#  - adj!(::X, ::X, ::X) -> X

"""
Some docs
"""
function optimise!(x::X, T::Real, RdR!, trace::TR=nothing; opts::OptOptions=OptOptions()) where {X, TR<:Union{Nothing, OptTrace}}
    # define functions to compute residuals with Optim.jl
    function fg!(F, G, x::OptVector)
        if G === nothing
            return RdR!(x.x, x.T)
        else
            R, _, dRdT = RdR!(G.x, x.x, x.T)
            G.T = dRdT
            return R
        end
    end

    # construct trace if one isn't provided
    t = isnothing(trace) ? OptTrace(x) : trace

    # print header
    if opts.verbose
        print_header(opts.io, t)
    end

    # perform optimisation using Optim.jl
    res = optimize(only_fg!(fg!), OptVector(x, T), opts.alg, genOptimOptions(opts, t))

    # unpack optimisation results
    x .= minimizer(res).x
    T  = minimizer(res).T

    return x, T, Dict("optim_output"=>res, "trace"=>t)
end
