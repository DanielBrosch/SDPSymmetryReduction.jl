"""
    Optimizer(optimizer_constructor; verbose = false, tol = 1e-6)

Return a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
optimizer that reduces a semidefinite program with [`admissible_subspace`](@ref)
and [`blockDiagonalize`](@ref), solves the reduced problem with an optimizer
created by `optimizer_constructor` and maps the solution back to the original
problem. It is only available once `MathOptInterface` is loaded, e.g., with
`import MathOptInterface` or `using JuMP`.

The problem should be in geometric conic form with a single semidefinite
constraint of side dimension `n`:
```
min/max     cᵀx
subject to  F₀ + ∑ᵢ xᵢ Fᵢ is positive semidefinite.
```
That is, in MathOptInterface's terminology, it should only contain free
variables, an affine objective and one
`MOI.VectorAffineFunction`-in-`MOI.PositiveSemidefiniteConeTriangle`
constraint. This is the form of the problems read from SDPA files. A problem
in standard conic form, with the variables constrained in one
`MOI.PositiveSemidefiniteConeTriangle` and `MOI.EqualTo` constraints, can be
solved in geometric conic form with `Dualization.dual_optimizer`.

The admissible subspace is computed for the problem
```
min         ⟨F₀, Z⟩
subject to  ⟨Fᵢ, Z⟩ = cᵢ,
            Z is positive semidefinite,
```
with `-cᵢ` instead of `cᵢ` for a maximization problem, which is the dual of
the problem up to the sign of its objective, and the semidefinite constraint is replaced by its orthogonal projection onto
the subspace. The reduced problem is in the same geometric conic form. Its
blocks of size at least 2 are `MOI.VectorAffineFunction`-in-
`MOI.PositiveSemidefiniteConeTriangle` constraints and its blocks of size 1 are
grouped in one `MOI.VectorAffineFunction`-in-`MOI.Nonnegatives` constraint.
The projection may make some variables redundant so the reduced problem only
keeps a subset of the variables whose projected matrices `Fᵢ` are linearly
independent.

The primal solution `x` is mapped back by solving, in the least squares
sense, `∑ᵢ xᵢ Fᵢ = S - F₀` where `S` is the projection of the matrix of the
semidefinite constraint at the solution of the reduced problem. The dual
solution `Z` of the semidefinite constraint is mapped back by giving the same
value to all the entries of a part, chosen such that `⟨Z, Pₚ⟩` is the value
of the dual of the reduced problem for the part `Pₚ`.

As the reduction is randomized and numerical, the equation
`∑ᵢ xᵢ Fᵢ = S - F₀` and the equality constraints `⟨Fᵢ, Z⟩ = cᵢ` of the dual
are checked once the solution is mapped back and a warning is displayed if
their residual is larger than `tol` times `1 +` the infinity norm of their
right-hand side.

Set `verbose = true` to print the progress of the reduction.

## Example

```julia
using JuMP, SDPSymmetryReduction
import Clarabel
model = Model(() -> SDPSymmetryReduction.Optimizer(Clarabel.Optimizer))
```
"""
function Optimizer end
