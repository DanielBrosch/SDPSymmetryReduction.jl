# # The Theta'-function of Erdos-Renyi graphs
# Let $q$ be an odd prime, and let $V = \mathrm{GF}(q)^3$ be a three dimensional 
# vector space over the finite field of order $q$. The set of one dimensional 
# subspaces, i.e. the projective plane, of $V$ is denoted by $\mathrm{PG}(2,q)$. 
# There are $q^2+q+1$ such subspaces, which are the vertices of the Erdos-Renyi 
# graph $\mathrm{ER}(q)$. Two vertices are adjacent if they are distinct and 
# orthogonal, i.e. for two representing vectors $x$ and $y$ we have $x^Ty=0$. We 
# are interested in the size of a maximum stable set of these graphs, 
# specifically upper bounds for this value. Note that these are not the equally named
# random graphs.

using LinearAlgebra #hide
q = 7
PG2q = vcat([[0, 0, 1]],
[[0, 1, b] for b = 0:q-1],
[[1, a, b] for a = 0:q-1 for b = 0:q-1])
Adj = [x' * y % q == 0 && x != y for x in PG2q, y in PG2q]
size(Adj)

# 
using Plots #hide
spy(Adj)


# ## The Theta'-function
# The Theta'-function $\vartheta(G)$ of a graph $G=(V,E)$ is such an upper bound, based on
# semidefinite programming:
#
# ```math
# \vartheta(G)\coloneqq \sup\{\langle X,J\rangle : \langle X,A\rangle = 0, X\succcurlyeq 0, X\geq 0\}.
# ```
#
# In vectorized standard form this is simply

N = length(PG2q)
C = ones(N^2)
A = vcat(vec(Adj)', vec(Matrix{Float64}(I, N, N))')
b = [0.0, 1.0];


# ## Determining the symmetry reduction 
# We can now apply the Jordan reduction method to the problem.
# First, we need to determine an (optimal) admissible subspace.

using SDPSymmetryReduction
P = admPartSubspace(C, A, b, true)
P.n

using Test #src 
@test P.n == 18 #src

# Running `admPartSubspace` returns a `Partition` object. `P.n` are the number of orbits (and thus
# variables), and `P.P` is a matrix with integer values from `1` trough `P.n`. Here, `P.P` looks like this
# (different color shades = different orbits): 

heatmap(reverse(P.P, dims=1)) #hide

# Now we can block-diagonalize the algebra (numerically)
blkD = blockDiagonalize(P, true);
@test sort(blkD.blkSizes) == [2,2,2,2,3] #src

# ## Building the reduced SDP
# Since `blkD.blks[i]` is the block-diagonalized image of `P.P .== i`,
# we obtain the new, symmetry reduced SDP by
using SparseArrays
PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)
newA = A * PMat
newB = b
newC = C' * PMat;

# ## Solving the SDP with JuMP and CSDP

using JuMP, CSDP
m = Model(CSDP.Optimizer)

## Initialize variables corresponding parts of the partition P
## >= 0 because the original SDP-matrices are entry-wise nonnegative
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
#
termination_status(m)
#
objective_value(m)

@test objective_value(m) ≈ 15.743402681126568 atol = 5 #src
