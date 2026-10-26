# Trace: a simple record of a search, passed as the callback of a method.

# the argument of the callbacks
_info(iter::Int, x::Orbit, R::Real, F, krylov::Vector{Float64}=Float64[], step::String="",
      trust::NamedTuple=(radius=NaN, length=NaN, ratio=NaN, rejected=0, trials=Tuple{Float64, Float64}[]);
      parameter::Real=NaN) =
    (iter        = iter,
     x           = x,
     res         = sqrt(2R),
     evaluations = counts(F.evaluations),
     krylov      = krylov,
     step        = step,
     trust       = trust,
     parameter   = parameter)

"""
    Trace()

History of a search, filled when passed as `callback` to [`LBFGS`](@ref) or
[`NewtonHookstep`](@ref). At the start and after every iteration it records the following fields.

- `iter`, the iteration number of the method.
- `res`, the residual norm `‖r‖`.
- `p`, the parameters `[log ω, c₁, …]`.
- `time`, the seconds elapsed since the first record.
- `evaluations`, the cumulative numbers of residuals, gradients, Jacobian actions, adjoint
  Jacobian actions and preconditioner applications of the system.
- `krylov`, for the hookstep, the relative residual of the linear model after each Arnoldi step
  of the iteration. It is empty for L-BFGS and at the start.
- `step`, for the hookstep, `"newton"` if the accepted step is the full Newton step, inside the
  trust region, or `"hook"` if it lies on its boundary. It is empty for L-BFGS and at the start.
- `radius`, `steplength`, `ratio` and `rejected`, for the hookstep. These are the trust-region
  radius with which the accepted step was computed (at the start, the initial radius), the length
  of that step in the metric of the preconditioner, the ratio of the actual to the predicted
  reduction of `½‖r‖²`, and the number of trial steps rejected before it. They are `NaN` and 0
  for L-BFGS.
- `trials`, for the hookstep, the radius and the ratio of each trial step rejected before the
  accepted one, in order. Each rejection divides the radius by 4. It is empty for L-BFGS.

The same trace can follow several calls to `solve!`, such as L-BFGS and then the hookstep. The
iteration numbers restart at zero with each call.

```julia
trace = Trace()
solve!(x, F, LBFGS(callback=trace))
solve!(x, F, NewtonHookstep(callback=trace))

trace.res # residual norms, in order
```
"""
struct Trace
           iter::Vector{Int}             # iteration number of the method
            res::Vector{Float64}         # residual norm ‖r‖
              p::Vector{Vector{Float64}} # parameters [log ω, c₁, …]
           time::Vector{Float64}         # seconds since the first record
    evaluations::Vector{NamedTuple}      # cumulative evaluations of the system
         krylov::Vector{Vector{Float64}} # relative residuals of the linear model, hookstep only
           step::Vector{String}          # "newton" or "hook", hookstep only
         radius::Vector{Float64}         # trust-region radius of the accepted step, hookstep only
     steplength::Vector{Float64}         # length of the accepted step in the metric of B
          ratio::Vector{Float64}         # actual over predicted reduction of ½‖r‖²
       rejected::Vector{Int}             # trial steps rejected before the accepted one
         trials::Vector{Vector{Tuple{Float64, Float64}}} # their radius and ratio ρ
             t₀::Base.RefValue{Float64}  # wall-clock time of the first record

    Trace() = new(Int[], Float64[], Vector{Float64}[], Float64[], NamedTuple[],
                  Vector{Float64}[], String[], Float64[], Float64[], Float64[], Int[],
                  Vector{Tuple{Float64, Float64}}[], Ref(0.0))
end

# record one iteration; never stops the search
function (trace::Trace)(info::NamedTuple)
    isempty(trace.time) && (trace.t₀[] = time())

    push!(trace.iter, info.iter)
    push!(trace.res, info.res)
    push!(trace.p, copy(info.x.p))
    push!(trace.time, time() - trace.t₀[])
    push!(trace.evaluations, info.evaluations)
    push!(trace.krylov, copy(info.krylov))
    push!(trace.step, info.step)
    push!(trace.radius, info.trust.radius)
    push!(trace.steplength, info.trust.length)
    push!(trace.ratio, info.trust.ratio)
    push!(trace.rejected, info.trust.rejected)
    push!(trace.trials, copy(get(info.trust, :trials, Tuple{Float64, Float64}[])))

    return false
end

Base.length(trace::Trace) = length(trace.res)
