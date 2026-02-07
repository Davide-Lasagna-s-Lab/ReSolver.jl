# Interface for the optimisation of a type using the residuals as objective.

# TODO: interface with OptimKit.jl and NLOpt.jl

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
    optimise!(x::X,
              T::Real,
              RdR!,
              state::OptState=OptState(x),
              opts::OptOptions=OptOptions())->(x, T, output)

Optimise an input `x` and period `T` for the goal of finding a periodic solution,
modifying `x` in-place in process.

# Arguments
- `x::X`: initial state-space loop to be optimised
- `T::Real`: initial period for the state-space loop to be optimised
- `RdR!`: objective functional that computes the residual and gradient, see
          [`Residual`](@ref) for how to construct this object
- `state`: state object that keeps track of useful variables during optimisation
- `opts`: optimisation option, see [`OptOptions`](@ref)

# Outputs
- `x::X`: final state-space loop after being optimised
- `T::Real`: period for the optimised state-space loop
- `output`: dictionary containing the optimisation trace and the output from Optim.jl
"""
function optimise!(x::X, T::Real, RdR!, state::OptState=OptState(x); opts::OptOptions=OptOptions()) where {X}
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
        print_header(opts.io)
    end

    # perform optimisation using Optim.jl
    res = optimize(only_fg!(fg!), OptVector(x, T), opts.alg, genOptimOptions(opts, state))

    # unpack optimisation results
    x .= minimizer(res).x
    T  = minimizer(res).T

    return x, T, Dict("optim_output"=>res, "final_state"=>state)
end
