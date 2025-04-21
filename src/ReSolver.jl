module ReSolver

using LinearAlgebra, Parameters, Printf, Optim

export Residual, optimise!, OptOptions

include("residual.jl")
include("optimvector.jl")
include("trace.jl")
include("options.jl")
include("callback.jl")
# include("optimise.jl")

end
