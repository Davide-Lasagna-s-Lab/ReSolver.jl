module ReSolver

using LinearAlgebra, Parameters, Printf#, Optim

export Residual, optimise!

include("residual.jl")
include("optimvector.jl")
# include("options.jl")
# include("trace.jl")
# include("callback.jl")
# include("optimise.jl")

end
