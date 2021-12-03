# SDPSymmetryReduction

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DanielBrosch.github.io/SDPSymmetryReduction.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DanielBrosch.github.io/SDPSymmetryReduction.jl/dev)
[![Build Status](https://github.com/DanielBrosch/SDPSymmetryReduction.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DanielBrosch/SDPSymmetryReduction.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DanielBrosch/SDPSymmetryReduction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DanielBrosch/SDPSymmetryReduction.jl)

# SDPSymmetryReduction

Numerically reduces semidefinite programming problems by exploiting their symmetry. Input is in vectorized standard form
```
sup/inf     dot(C,x)
subject to  Ax = b,
            Mat(x) is positive semidefinite/doubly nonnegative,
```
where `C` and `b` are vectors and `A` is a matrix.

## Installation
Simply run
```julia
pkg> add SDPSymmetryReduction  # Press ']' to enter the Pkg REPL mode.
```

## Main use
The function `admPartSubspace` determines an optimal admissible partition subspace for the problem. This is done using a randomized Jordan-reduction algorithm, and it returns a Jordan algebra (closed under linear combinations and squaring). SDPs can be restricted to such a subspace without changing their optimal value.

The function `blockDiagonalize` determines a block-diagonalization of a (Jordan)-algebra given by a partition `P` using a randomized algorithm.

For more details, see the [documentation](https://DanielBrosch.github.io/SDPSymmetryReduction.jl/stable).

## Example: Theta'-function
Let `Adj` be an adjacency matrix of an (undirected) graph `G`. Then the Theta'-function of the graph is given by
```
sup         dot(J,X)
subject to  dot(Adj,X) = 0,
            dot(I,X) = 1,
            X is positive semidefinite,
            X is entry-wise nonnegative,
```
where `J` is the all-ones matrix, and `I` the identity. Then we can exploit the symmetry of the graph and calculate this function by
```julia
using SDPSymmetryReduction
using LinearAlgebra, SparseArrays
using JuMP, MosekTools

# Theta' SDP
N = size(Adj,1)
C = ones(N^2)
A = hcat(vec(Adj), vec(Matrix(I, N, N)))'
b = [0, 1]

# Find the optimal admissible subspace (= Jordan algebra)
P = admPartSubspace(C, A, b, true)

# Block-diagonalize the algebra
blkD = blockDiagonalize(P, true)

# Calculate the coefficients of the new SDP
PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)
newA = A * PMat
newB = b
newC = C' * PMat

# Solve with optimizer of choice
m = Model(Mosek.Optimizer)

# Initialize variables corresponding parts of the partition P
# >= 0 because the original SDP-matrices are entry-wise nonnegative
x = @variable(m, x[1:P.n] >= 0)

@constraint(m, newA * x .== newB)
@objective(m, Max, newC * x)

psdBlocks = sum(blkD.blks[i] .* x[i] for i = 1:P.n)
for blk in psdBlocks
    if size(blk, 1) > 1
        @constraint(m, blk in PSDCone())
    else
        @constraint(m, blk .>= 0)
    end
end

optimize!(m)

@show termination_status(m)
@show value(newC * x)
```
There are more examples in the [documentation](https://DanielBrosch.github.io/SDPSymmetryReduction.jl/stable).

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
