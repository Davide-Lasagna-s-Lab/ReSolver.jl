module ReSolver

using LinearAlgebra

import Parameters: @with_kw
import Printf: @printf

import Optim: LineSearches,
              optimize,
              only_fg!,
              minimizer,
              Options,
              LBFGS,
              GradientDescent,
              ConjugateGradient,
              AbstractOptimizer

export GradientDescent, LBFGS, ConjugateGradient, LineSearches

export Residual, OptOptions, optimise!

include("residual.jl")
include("optvector.jl")
include("state.jl")
include("options.jl")
include("callback.jl")
include("optimise.jl")


# define a few small test systems for demonstration purposes
export ToySystems

include("ToySystems/ToySystems.jl")

end
