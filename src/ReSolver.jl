module ReSolver

import LinearAlgebra: dot, norm
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

include("residual.jl")
include("optvector.jl")
include("trace.jl")
include("options.jl")
include("callback.jl")
include("optimise.jl")


# TODO: change test dependency paradigm

# define a few small test systems for demonstration purposes
export ToySystems

include("ToySystems/ToySystems.jl")

end
