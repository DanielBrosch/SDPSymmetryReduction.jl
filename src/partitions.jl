"""
    Partition

A partition subspace. `P.n` is the number of parts, and `P.P` an integer matrix defining the basis elements.

`Partition` stores a partition of `1:n × 1:n in a single matrix
Entries of P should always be 1,…,n
"""
struct Partition{T<:Integer}
    n::Int # Number of parts
    P::Matrix{T} # Matrix with entries 1,...,n
end

Partition(args...) = Partition{UInt16}(args...)

"""
    Partition(M::AbstractMatrix)

Create a partition from the unique entries of `M`.
"""
function Partition{T}(M::AbstractMatrix) where {T}
    l = 0
    d = Dict(zero(eltype(M)) => T(l))
    res = zeros(T, size(M))
    for (v, idx) in zip(M, eachindex(res))
        k = get!(d, v, T(l + 1))
        l = ifelse(k == l + 1, k, l)
        res[idx] = k
    end

    return Partition{T}(l, res)
end

function Partition{T}(M::AbstractMatrix{<:Integer}) where {T}
    M_vals = unique(M)
    @assert 0 ≤ minimum(M_vals)
    vals = zeros(Int, maximum(M_vals) + 1) # shift by one to accomodate 0
    for (i, v) in pairs(M_vals)
        vals[v+1] = i # vals[1] = 0, if present
    end
    res = zeros(T, size(M))
    for (v, idx) in zip(M, eachindex(res))
        res[idx] = vals[v+1]
    end
    return Partition{T}(length(M_vals), res)
end

function refine(P1::Partition{T}, P2::Partition{S}) where {T,S}
    TT = promote_type(T, S)
    M = zeros(TT, size(P1.P))
    M .= P2.P # we really want to widen BEFORE addition
    M .+= P1.P .* (P2.n + 1)
    return Partition{T}(M)
end

randomize(P::Partition) = randomize(Float64, P)
randomize(::Type{T}, P::Partition) where {T} = randomize!(Matrix{T}(undef, size(P.P)), P)
randomize!(M::AbstractMatrix, P::Partition) =
    fill!(M, P; values=rand(eltype(M), P.n + 1))

"""
    fill!(M::AbstractMatrix, P::Partition; values::AbstractVector)
Fill matrix `M` with values from `values`, according to partition `P`.
"""
function Base.fill!(M::AbstractMatrix, P::Partition; values::AbstractVector)
    values[1] = zero(eltype(values))
    @inbounds if eltype(values) <: Real
        for idx in eachindex(P.P, M)
            M[idx] = values[P.P[idx]+1]
        end
    else
        M .= zero(eltype(M))
        for idx in eachindex(IndexCartesian(), P.P, M)
            v = values[P.P[idx]+1]
            t = Tuple(idx)
            M[t...] += v
            M[reverse(t)...] += v'
        end
    end
    return M
end

