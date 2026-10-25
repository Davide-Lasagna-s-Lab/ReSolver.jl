using Documenter, PeriodicBSplines

makedocs(sitename="PeriodicBSplines.jl", modules=[PeriodicBSplines],
         format=Documenter.HTML(prettyurls=false, mathengine=Documenter.MathJax3()),
         pages=["Home" => "index.md", "Theory" => "theory.md", "API" => "api.md"],
         checkdocs=:exports)
