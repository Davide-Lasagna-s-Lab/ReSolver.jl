# Interface for the optimisation of a type using the residuals as objective.

export optimise!

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
            return RdR!(G, x.x, x.T)
        end
    end

    # print header
    if opts.verbose
        print_header(opts.io, trace)
    end

    # perform optimisation using Optim.jl
    return optimize(Optim.only_fg(fg!), OptVector(x, T), genOptimOptions(opts))
end
