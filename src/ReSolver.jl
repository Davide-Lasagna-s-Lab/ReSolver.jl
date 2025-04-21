module ReSolver

using LinearAlgebra, Parameters, Printf, Optim

include("residual.jl")
include("optvector.jl")
include("trace.jl")
include("options.jl")
include("callback.jl")
# include("optimise.jl")

end
