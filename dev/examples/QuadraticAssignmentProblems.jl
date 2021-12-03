# # Symmetry reducting a strong relaxation of the quadratic assigment problem
# Here, we are going to show how to load a QAP from QABLib, formulate 
# a strong semidefinite relaxation of it, symmetry reduce it, and finally solve it.

# ## Quadratic assigment problems
# QAPs are given by two quadratic matrices $A$ and $B$. The objective is to permute 
# the rows and columns of $B$, such that the inner product between the matrices is 
# minimized.
#
# ``\mathrm{QAP}(A,B) = \min_{\phi\in S_n} \sum_{i,j=1}^n a_{ij}b_{\phi(i)\phi(j)}``
#
# QAPs are notoriously hard to solve exactly, but there exist strong polynomial time
# relaxations, such as the following semidefinite programming relaxation:
#
# ```math
# \begin{aligned}
# \min\enspace & \langle B\otimes A ,Y\rangle\\
# \mathrm{s.t.}\enspace & \langle I_n\otimes E_{jj},Y\rangle=1 \text{ for }j\in [n],\\
# & \langle E_{jj}\otimes I_n,Y\rangle=1 \text{ for }j\in [n],\\
# & \langle I_n\otimes (J_n-I_n)+(J_n-I_n)\otimes I_n,Y\rangle =0, \\
# & \langle J_{n^2},Y\rangle = n^2,\\
# & Y\in D^{n^2},
# \end{aligned}
# ```
#
# But in practice this relaxation is often too big to be solved directly.


# ## Loading the data of a QAP
using SparseArrays, LinearAlgebra

file = joinpath(@__DIR__, "esc16j.dat")
data = open(file) do file
    read(file, String)
end
data = split(data, [' ', '\n', '\r'], keepempty = false)

n = parse(Int64, data[1])
A = zeros(Int64, n, n)
B = zeros(Int64, n, n)

pos = 2
for x = 1:n
    for y = 1:n
        A[x, y] = parse(Int64, data[pos])
        global pos += 1
    end
end
for x = 1:n
    for y = 1:n
        B[x, y] = parse(Int64, data[pos])
        global pos += 1
    end
end


# ## Building the SDP (in vectorized standard form)
n = size(A, 1)

## Objective
CPrg = sparse(kron(B, A))

In = sparse(I, n, n)
Jn = ones(n, n)

## Vectorised constraint matrices as rows of large matrix APrg
APrg = spzeros(2n + 1, n^4)
bPrg = zeros(2n + 1)
currentRow = 1

for j = 1:n
    Ejj = spzeros(n, n)
    Ejj[j, j] = 1.0
    APrg[currentRow, :] = vec(kron(In, Ejj))
    bPrg[currentRow] = 1
    global currentRow += 1
    ## Last constraint is linearly dependent on others
    if (j < n)
        APrg[currentRow, :] = vec(kron(Ejj, In))
        bPrg[currentRow] = 1
        global currentRow += 1
    end
end

APrg[currentRow, :] = vec(kron(In, Jn - In) + kron(Jn - In, In))
bPrg[currentRow] = 0
currentRow += 1
APrg[currentRow, :] = vec(ones(n^2, n^2))
bPrg[currentRow] = n^2

CPrg = sparse(vec(0.5 * (CPrg + CPrg')));

# ## Symmetry reducing the SDP 

# We first determine an optimal admissible partition subspace
using SDPSymmetryReduction
P = admPartSubspace(CPrg, APrg, bPrg, true)
P.n
# And then we block-diagonalize it 
blkD = blockDiagonalize(P, true);

# ## Determining the coefficients of the reduced SDP 
PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)
newA = APrg * PMat
newB = bPrg
newC = CPrg' * PMat;

## Removing linearly dependent constraints #src
# using RowEchelon  #src
# T = rref!(Matrix(hcat(newA,newB))) #src
# r = rank(T) #src
# newA = T[1:r,1:end-1] #src
# newB = T[1:r,end] #src
# length(newB) #src

# ## Solving the reduced SDP with JuMP and CSDP

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