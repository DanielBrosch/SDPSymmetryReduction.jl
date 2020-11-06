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

## Determining an admissible subspace
The function `admPartSubspace` determines an optimal admissible partition subspace for the problem. This is done using a randomized Jordan-reduction algorithm, and it returns a Jordan algebra (closed under linear combinations and squaring). SDPs can be restricted to such a subspace without changing their optimal value.

Given `C,A` and `b`, `admPartSubspace(C,a,b)` returns a `Partition P` with `P.n` giving the number of parts of the partition, and `P.P` returning an integer valued matrix (same size at `x` in matrix form) with entries `1,...,n` defining the partition.

## Block-diagonalizing a Jordan-algebra
The function `blockDiagonalize` determines a block-diagonalization of a (Jordan)-algebra given by a partition `P` using a randomized algorithm.

`blockDiagonalize(P)` returns a real block-diagonalization `blkd`, if it exists, otherwise `nothing`.
* `blkd.blkSizes` returns an integer array of the sizes of the blocks.
* `blkd.blks` returns an array of length `P.n` containing arrays of (real) matrices of sizes `blkd.blkSizes`. I.e. `blkd.blks[i]` is the image of the basis element `P.P .== i`.

`blockDiagonalize(P; complex = true)` returns the same, but with complex valued matrices, and should be used if no real block-diagonalization was found. To use the complex matrices practically, remember that a Hermitian matrix `A` is positive semidefinite iff `[real(A) -imag(A); imag(A) real(A)]` is positive semidefinite.

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
There are more examples in the folder `examples`:
* `ReduceAndSolveJuMP.jl`: A more advanced function for fully reducing and solving SDPs with JuMP and Mosek, including support for complex block-diagonalizations.
* `ErdosRenyiThetaFunction.jl`: A full example calculating the Theta'-function of Erdos-Renyi graphs.
* `QuadraticAssignmentProblems.jl`: Loads a QAP in QAPLib format and then reduces and solves a strong doubly nonnegative relaxation of it.
