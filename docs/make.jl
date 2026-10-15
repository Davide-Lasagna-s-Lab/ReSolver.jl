# Build the documentation:
#
#     julia --project=docs docs/make.jl
#
# On GitHub Actions the Documentation workflow builds it and deploydocs pushes it to the gh-pages
# branch, dev/ from main and stable/ and v0.x.y/ from the tags. Run locally, deploydocs does nothing.
#
# The figures are in docs/src/assets/<example>/, one folder per example, where the scripts of the
# examples write them and the pages link them.
#
# The research notes on Kolmogorov flow, and their figures, live with the manuscript of the
# space-time Newton–Krylov paper, not in this repository.

using Documenter
using ReSolver

# ---- pages ----
makedocs(sitename = "ReSolver.jl",
         authors  = "Davide Lasagna",
         modules  = [ReSolver],
         format   = Documenter.HTML(mathengine     = Documenter.KaTeX(),
                                    prettyurls     = true,
                                    canonical      = "https://davide-lasagna-s-lab.github.io/ReSolver.jl/stable/",
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
                                    "examples/methods.md",
                                    "examples/kolmogorov_flow.md"],
                     "API"      => "api.md"],
         checkdocs = :exports,
         warnonly  = [:missing_docs, :cross_references])

# ---- deployment, from the Documentation workflow ----
deploydocs(repo      = "github.com/Davide-Lasagna-s-Lab/ReSolver.jl.git",
           devbranch = "main")
