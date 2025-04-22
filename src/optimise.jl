# Interface for the optimisation of a type using the residuals as objective.

export optimise!

# TODO: the docs
# TODO: interface with OptimKit.jl

# ~~~ Assumed interface for type X ~~~
#  - eltype(::X) -> Type
#  - similar(::X) -> X
#  - dot(::X, ::X) -> Number
#  - norm(::X) -> Non-negative Real
#  - broadcasting between variables of type X
#  - dds!(::X, ::X) -> X
#  - rhs!(::X, ::X) -> X
#  - adj!(::X, ::X, ::X) -> X

"""
Some docs
"""
function optimise!(x::X, T::Real, RdR!::Residual{X}, trace::TR=nothing; opts::OptOptions=OptOptions()) where {X, TR<:Union{Nothing, OptTrace}}
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

    # print header
    if opts.verbose
        print_header(opts.io, trace)
    end

    # perform optimisation using Optim.jl
    res = optimize(only_fg!(fg!), OptVector(x, T), opts.alg, genOptimOptions(opts, trace))

    # unpack optimisation results
    x .= minimizer(res).x
    T  = minimizer(res).T

    return x, T, Dict("optim_output"=>res)
end
