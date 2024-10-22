module SDPSymmetryReduction

using LinearAlgebra
using Random
using SparseArrays
using DataStructures

export Partition, admPartSubspace, blockDiagonalize
include("utils.jl")
include("compat.jl")
include("partitions.jl")
include("eigen_decomposition.jl")
include("diagonalize.jl")


"""
    unSymmetrize(P::Partition)

WL algorithm to "un-symmetrize" the Jordan algebra corresponding to `P`.
"""
function unSymmetrize(P::Partition)
    P = deepcopy(P)
    dim = P.n
    it = 0
    # Iterate until converged
    while true
        it += 1
        randomizedP1 = rndPart(P)
        randomizedP2 = rndPart(P)
        P2 = randomizedP1 * randomizedP2
        P2 = roundMat(P2)
        P = coarsestPart(P, part(P2))
        # Check if converged
        if dim == P.n
            break
        end
        dim = P.n
    end
    return P
end
unSymmetrize(q::Matrix) = unSymmetrize(Partition(q))
export unSymmetrize


end
