# Build the documentation:
#
#     julia --project=docs docs/make.jl
#
# The figures are produced by the scripts in examples/, in their figures/ folders, and copied into
# docs/src/assets.

using Documenter
using ReSolver

# ---- figures of the examples ----
assets = joinpath(@__DIR__, "src", "assets")
mkpath(assets)

for (folder, prefix) in (("kuramoto_sivashinsky", "ks"), ("lorenz", "lorenz"), ("metric", "metric"))
    dir = joinpath(@__DIR__, "..", "examples", folder, "figures")
    for file in filter(endswith(".png"), readdir(dir))
        cp(joinpath(dir, file), joinpath(assets, "$(prefix)_$(file)"); force=true)
    end
end

# ---- pages ----
makedocs(sitename = "ReSolver.jl",
         authors  = "Davide Lasagna",
         modules  = [ReSolver],
         format   = Documenter.HTML(mathengine = Documenter.KaTeX(),
                                    prettyurls = true,
                                    canonical  = "https://davide-lasagna-s-lab.github.io/ReSolver.jl/",
                                    size_threshold = 400_000),
         pages    = ["Home"     => "index.md",
                     "Usage"    => "usage.md",
                     "Theory"   => ["theory/formulation.md",
                                    "theory/optimisation.md",
                                    "theory/root_finding.md",
                                    "theory/preconditioning.md"],
                     "Examples" => ["examples/lorenz.md",
                                    "examples/kuramoto_sivashinsky.md",
                                    "examples/ks_symmetric.md",
                                    "examples/methods.md"],
                     "API"      => "api.md"],
         checkdocs = :exports,
         warnonly  = [:missing_docs, :cross_references])
