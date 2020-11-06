using SDPSymmetryReduction
using LinearAlgebra, SparseArrays
using JuMP, MosekTools

## Calculating the Theta'-function of Erdos-Renyi graphs
q = 31

# Generating the adjacency matrix of ER(q)
PG2q = vcat([[0, 0, 1]], [[0, 1, b] for b = 0:q-1], [[1, a, b] for a = 0:q-1 for b = 0:q-1])
Adj = [x' * y % q == 0 for x in PG2q, y in PG2q]
Adj[diagind(Adj)] .= 0

# Theta' SDP
N = length(PG2q) # = q^2+q+1
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
