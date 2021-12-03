```@meta
CurrentModule = SDPSymmetryReduction
```

# SDPSymmetryReduction.jl
*A symmetry reduction package for Julia.*

Documentation for [SDPSymmetryReduction](https://github.com/DanielBrosch/SDPSymmetryReduction.jl).

This package provides functions to determine a symmetry reduction of an [SDP](https://en.wikipedia.org/wiki/Semidefinite_programming) numerically using the Jordan Reduction method.

It assumes that the problem is given in vectorized standard form
```
sup/inf     dot(C,x)
subject to  Ax = b,
            Mat(x) is positive semidefinite/doubly nonnegative,
```
where `C` and `b` are vectors and `A` is a matrix.

## Examples
```@contents
Pages = ["examples/ErdosRenyiThetaFunction.md", "examples/QuadraticAssignmentProblems.md", "examples/ReduceAndSolveJuMP.md"]
Depth = 1
```

## Documentation

```@autodocs
Modules = [SDPSymmetryReduction]
Private = false
```
