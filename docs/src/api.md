# API

## Unknowns and system

```@docs
Orbit
ReSolver.frequency
System
```

## Methods

```@docs
solve!
LBFGS
NewtonHookstep
```

## Tracing

```@docs
Trace
```

## Preconditioners

```@docs
ReSolver.precondition!
ReSolver.precondition_adjoint!
ReSolver.relinearise!
```

## Residual, gradient and Jacobian

These functions are used by the methods and are not exported. They are useful to check a system.

```@docs
ReSolver.residual!
ReSolver.objective
ReSolver.gradient!
ReSolver.jacobian!
ReSolver.jacobian_adjoint!
```
