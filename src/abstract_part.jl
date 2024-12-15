"""
    AbstractPartition
Abstract type representing partition subspace.
A partition subspace is a partition of `1:n × 1:n` into disjoint subsets.
The first set has a special meaning and should be preserved during refines.

Necessary methods are:
 * `(P::AbstractPartition)(M::AbstractMatrix)` - construct a partition subspace
    from a matrix of numerical values.
 * `dim(p::AbstractPartition)` - the dimension of the partition
 * `Base.size(p::AbstractPartition, args...)` - follows the `Base.size` for
    `AbstractArrays`
 * `Base.fill!(M::AbstractMatrix, p::AbstractPartition; values)` - fills `M`
    with `values` according to partition `p`.
 * `refine!(p::AP, q::AP) where {AP<:AbstractPartition}` - find coarsest common
    refinement of `p` and `q` and return it as a `AP`.
"""
abstract type AbstractPartition end

"""
    dim(ap::AbstractPartition)
The dimension of the partition subspace `ap`.

# !!! note
#     The `0`-set (the first subset) has a special meaning and is not accounted
#     for the calculation.

# Examples
```julia
julia> q = SR.Partition(Float64[
           1 1 0;
           1 0 5;
           0 3 3
       ]);

julia> SR.dim(q)
3

julia> p = SR.Partition([
           1 1 2;
           1 2 5;
           2 3 3
       ]);

julia> SR.dim(p)
4

```
"""
function dim end

"""
    fill!(M::AbstractMatrix, p::AbstractPartition; values)
Fill `M` with `values` according to partition `p`.

The fill will to preserve the `0`-set.

# Examples
```julia
julia> q = SR.Partition(Float64[
           1 1 0;
           1 0 5;
           0 3 3
       ]);

julia> fill!(zeros(3,3), q, values=[-1, sqrt(2), π])
3×3 Matrix{Float64}:
 -1.0  -1.0      0.0
 -1.0   0.0      3.14159
  0.0   1.41421  1.41421
```
"""
function Base.fill!(M::AbstractMatrix, ap::AbstractPartition; values::AbstractVector) end

"""
    randomize([T=Float64, ]P::AbstractPartition)
Return a `Matrix{T}` filled with random values according to partition subspace `P`.

`0`-set of `P` will be mapped to `zero(T)`.

# Examples
```julia
julia> q = SR.Partition([
           1 1 0;
           1 0 5;
           0 3 3
       ]);

julia> SR.randomize(q)
3×3 Matrix{Float64}:
 0.916099  0.916099  0.0
 0.916099  0.0       0.0117013
 0.0       0.052362  0.052362

```
"""
randomize(P::AbstractPartition) = randomize(Float64, P)
randomize(::Type{T}, P::AbstractPartition) where {T} =
    randomize!(Matrix{T}(undef, size(P)), P)

"""
    randomize!(M::AbstractMatrix, P::AbstractPartition)
Randomize `M` in-place according to partition subspace `P`.

See also [`fill!(::AbstractMatrix, ::AbstractPartition)`](@ref Base.fill!(::AbstractMatrix, ::AbstractPartition)).
"""
function randomize!(M::AbstractMatrix, P::AbstractPartition)
    values = rand(eltype(M), dim(P))
    return fill!(M, P; values=values)
end

"""
    refine!(p::AP, q::AP) where AP<:AbstractPartition
Find the coarsest common refinement of partitions `p` and `q` modifying `p` in-place.
"""
function refine! end

