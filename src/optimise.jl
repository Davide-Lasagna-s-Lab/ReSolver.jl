# Interface for the optimisation of a type using the residuals as objective.

# TODO: multiple modes for the optimisation package to use (Optim.jl or OptimKit.jl)

function optimise!(x::X, RdR!::Residual{X}, opts) where {X}
    throw(error("I'm meant to do something!"))
end
