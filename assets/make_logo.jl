# Draws assets/logo.svg: a closed loop that does not satisfy the equations, the initial guess of the
# variational method, converging through a smoother loop to the periodic orbit, a smooth closed loop in
# Julia colours.
#
#     julia assets/make_logo.jl

using Printf


# ---- the orbit, a closed loop with some structure ----

# point of the orbit at the phase θ
orbit(θ) = (100 * cos(θ) + 12 * cos(2θ) - 4 * sin(3θ), 68 * sin(θ) + 9 * sin(3θ) - 6 * cos(2θ))

# a closed loop of the iteration: the orbit with a periodic perturbation of size e, whose high harmonics
# fade faster than its low ones as the iteration converges
function loop(e; n=360)
    θs = range(0, 2π; length=n + 1)[1:end - 1]
    return map(θs) do θ
        x, y = orbit(θ)
        (x + e * (16 * sin(2θ + 5.6) + e * 4 * sin(7θ + 0.4)), y + e * (14 * cos(3θ + 5.1) + e * 4 * cos(6θ + 1.1)))
    end
end

path(pts; close=false) = "M" * join([@sprintf("%.1f %.1f", p...) for p in pts], " L") * (close ? " Z" : "")

guess        = loop(1.0)
intermediate = loop(0.45)

# ---- the orbit in four arcs, one per Julia colour, and an arrow along it ----
COLORS = ["#4063d8", "#9558b2", "#cb3c33", "#389826"]
arcs   = [path([orbit(θ) for θ in range(2π * (k - 1) / 4, 2π * k / 4; length=80)]) for k in 1:4]
arrow  = path([orbit(θ) for θ in range(1.30π, 1.46π; length=12)])


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 300" role="img" aria-labelledby="title desc">
  <title id="title">ReSolver.jl logo</title>
  <desc id="desc">A rough closed loop, the initial guess, converges through a smoother loop to a smooth closed periodic orbit drawn in Julia colors, beside the ReSolver.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.10"/>
    </filter>
    <marker id="flow-arrow" viewBox="0 0 12 10" refX="6" refY="5" markerWidth="4.4" markerHeight="3.7" orient="auto">
      <path d="M0 0 L12 5 L0 10 Z" fill="#242936"/>
    </marker>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="912" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- Left illustration: closed loops converging to a periodic orbit -->
  <g transform="translate(212 150)" filter="url(#soft)">
    <path d="$(path(guess; close=true))" fill="none" stroke="#98a2b3" stroke-width="3" stroke-dasharray="7 6" stroke-linecap="round"/>
    <path d="$(path(intermediate; close=true))" fill="none" stroke="#c3b1d6" stroke-width="3.5" stroke-dasharray="11 5" stroke-linecap="round"/>
    $(join(["<path d=\"$(a)\" fill=\"none\" stroke=\"$(c)\" stroke-width=\"7\" stroke-linecap=\"round\"/>" for (a, c) in zip(arcs, COLORS)], "\n    "))
    <path d="$arrow" fill="none" stroke="#242936" stroke-width="4" stroke-linecap="round" marker-end="url(#flow-arrow)"/>
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(372 96)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="58" font-weight="820" fill="#222734" letter-spacing="0">ReSolver.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Variational Newton–Krylov periodic orbit solver,</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">with L-BFGS descent.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
