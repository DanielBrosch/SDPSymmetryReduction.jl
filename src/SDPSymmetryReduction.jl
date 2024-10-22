module SDPSymmetryReduction

using LinearAlgebra
using Random
using SparseArrays
using DataStructures

export Partition, admPartSubspace, blockDiagonalize, unSymmetrize

include("utils.jl")
include("compat.jl")
include("partitions.jl")
include("eigen_decomposition.jl")
include("diagonalize.jl")

end
