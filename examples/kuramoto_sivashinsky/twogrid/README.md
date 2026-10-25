# Two-grid preconditioner: study on Kuramoto–Sivashinsky

The scripts behind the report *A two-grid preconditioner for space-time Newton–Krylov searches of
periodic orbits*. They compare Newton–hookstep searches with ReSolver's diagonal preconditioner
alone and with `TwoGrid` (`src/twogrid.jl`), on Kuramoto–Sivashinsky with L = 39 in the odd
subspace, for periods from 25 to 1600 (with the diagonal preconditioner alone, up to 800).

| file | content |
|---|---|
| `common.jl` | test cases (`prime_case`, `repeated_start`) and `run_newton`, which runs one search and returns its costs |
| `smoothing_map.jl` | amplification of one smoothing step on every space-time Fourier mode |
| `study.jl` | the scans: `T` (cost against the period; `Ttwogrid`, `Tplain` for one part), `r` (truncation level in space and time), `cycle` (cycles and time derivative of the coarse operator) |
| `figures.py` | the figures, from `results/*.txt` |
| `explore/` | exploratory runs quoted in the report, with logs |
| `data/` | converged orbits and starting points from the earlier study |
| `results/` | one line per search (columns in `study.jl`), and the logs |

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. smoothing_map.jl
julia --project=. study.jl T
julia --project=. study.jl r
julia --project=. study.jl cycle
python3 figures.py
```

The environment uses ReSolver from this repository and BorderedMatrices.jl from GitHub, which
enables the banded coarse solver (`solver=:banded`).
