# Interface for the optimisation of a type using the residuals as objective.

"""
Some docs
"""
function optimise!(x::X, RdR!::Residual{X}, trace::T=nothing; opts::OptOptions=OptOptions()) where {X, T<:Union{Nothing, OptimTrace}}
    # define functions to compute residuals with Optim.jl
    function fg!(F, G, a::X) where {X}
        if G === nothing
            return RdR!(a)
        else
            return RdR!(G, a)
        end
    end

    # print header
    if opts.verbose
        print_header(opts.io, trace)
    end

    # perform optimisation using Optim.jl
    return optimize(Optim.only_fg(fg!), x, genOptimOptions(opts))
end
