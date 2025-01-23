"""
    Partition

A partition subspace stored internally as matrix of integers from `0` to `dim(P)`.
"""
mutable struct Partition{T<:Integer} <: AbstractPartition
    nparts::Int # Number of parts
    matrix::Matrix{T} # Matrix with entries 1,...,n
end

Partition(args...) = Partition{UInt32}(args...)

dim(p::Partition) = p.nparts
Base.size(p::Partition, args...) = size(p.matrix, args...)

Base.:(==)(p::Partition, q::Partition) =
    dim(p) == dim(q) && p.matrix == q.matrix

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
    @assert 0 ≤ first(M_vals)
    vals = zeros(Int, maximum(M_vals) + 1) # to accomodate for 0 if it exists
    dim = 0
    for v in M_vals
        iszero(v) && continue # to preserve 0
        dim += 1
        vals[v+1] = dim
    end
    res = zeros(T, size(M))
    for (v, idx) in zip(M, eachindex(res))
        res[idx] = vals[v+1]
    end
    return Partition{T}(dim, res)
end

function __sort_unique!(P::Partition)
    M_vals = unique(P.matrix)
    @assert 0 ≤ first(M_vals)
    # vals = Dict{Int, Int}(1 => 0) # doesn't seem worth it, even with gaps
    vals = zeros(Int, maximum(M_vals) + 1) # to accomodate for 0 if it exists
    dim = 0
    for v in M_vals
        iszero(v) && continue # to preserve 0
        dim += 1
        vals[v+1] = dim
    end
    for (idx, v) in pairs(P.matrix)
        P.matrix[idx] = vals[v+1]
    end
    P.nparts = dim
    return P
end

function refine!(P1::Partition{T}, P2::Partition{S}) where {T,S}
    P1.matrix .+= P2.matrix .* (dim(P1) + 1)
    P1 = __sort_unique!(P1)
    return P1
end

function Base.fill!(M::AbstractMatrix, P::Partition; values::AbstractVector)
    @assert length(values) == dim(P)
    for idx in eachindex(P.matrix, M)
        k = P.matrix[idx]
        M[idx] = iszero(k) ? zero(eltype(M)) : values[k]
    end
    return M
end

function admissible_subspace(
    C::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    verbose::Bool=false,
    atol=Base.rtoldefault(real(T)),
) where {T<:AbstractFloat}
    return admissible_subspace(Partition{UInt16}, C, A, b; verbose=verbose, atol=atol)
end

"""
    admissible_subspace([Partition{UInt16},] C, A, b[; verbose, atol])
Return the optimal admissible partition subspace for the semidefinite problem

> ```
>   minimize: ⟨Cᵀ, x⟩
> subject to: A·x = b
>             Mat(x) ⪰ 0
> ```

The problem can be restricted to the subspace without changing its optimal value.

The partition is found by a Jordan-reduction algorithm saturating with __random__
squares. With probability 1 the returned subspace is a Jordan algebra
(i.e. closed under linear combinations and squaring).
See Section 5.2 of Permenter thesis.

# Output
* A `Partition` subspace `P`.
"""
function admissible_subspace(
    ::Type{Part},
    C::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    verbose::Bool=false,
    atol=Base.rtoldefault(real(T)),
) where {Part<:AbstractPartition,T<:AbstractFloat}
    n = isqrt(length(C))
    @assert n^2 == length(C)
    # temporary values for re-use
    tmp = Vector{T}(undef, length(C))
    X = Matrix{Float64}(undef, n, n)
    X² = Matrix{Float64}(undef, n, n)

    projLᵖ! = let A′ = A', A′qr = qr(A′)
        (tmp, v) -> project_colspace!(tmp, v, A′, Afact=A′qr)
    end

    # notation according to Brosch
    CL = let c = Vector(C) # we own `c` from now on
        c .-= projLᵖ!(tmp, c)
        c = _clamp_round!(c, atol=atol)
        c = _symmetrize!(c, n)
        reshape(c, n, n)
    end

    # Krylov.craig is equivalent to A\x
    X₀Lᵖ = let (X, _) = Krylov.craig(A, b) # we own `X`
        X = _symmetrize!(X, n)
        X = (tmp = projLᵖ!(tmp, X); copyto!(X, tmp))
        X = _clamp_round!(X, atol=atol)
        reshape(X, n, n)
    end

    # Initialize S as the span of the initial two elements
    S = Part(CL)
    S = refine!(S, Part(X₀Lᵖ))

    maximal_dimension = (n^2 + n) ÷ 2
    current_dimension = initial = dim(S)
    verbose && @info "Starting the reduction. Dimensions:" maximal = maximal_dimension initial = initial

    it = 0
    # Iterate until converged
    while current_dimension < maximal_dimension
        it += 1
        verbose && @debug "Iteration $it, Current dimension: $current_dimension"

        # Add a random projection to S
        X = randomize!(X, S)
        let x = vec(X) # x in here shares memory with X!
            x .-= projLᵖ!(tmp, x)
            x = _clamp_round!(x, atol=atol)
        end
        S = refine!(S, Part(X))

        if current_dimension != dim(S)
            X = randomize!(X, S)
        end


        # Add random square
        X² = mul!(X², X, X)
        X² = _clamp_round!(X², atol=atol)
        S = refine!(S, Part(X²))

        # with probability 1 a random square does not refine S
        # only when S is already closed under taking squares,
        # i.e. it a Jordan subalgebra, at least for special ones.
        # See Theorem 5.2.3 in Permenter thesis
        if current_dimension == dim(S) # converged
            break
        end

        current_dimension = dim(S)
    end

    verbose &&
        @info "Minimal admissible subspace converged in $it iterations at dimension:" final = dim(S)
    return S
end

"""
    desymmetrize(P::AbstractPartition[; verbose, atol])

WL algorithm to "desymmetrize" the Jordan algebra corresponding to `P`.
"""
function desymmetrize(P::AbstractPartition; verbose=false, atol=Base.rtoldefault(Float64))
    # preallocate
    X = Matrix{Float64}(undef, size(P))
    Y = Matrix{Float64}(undef, size(P))
    XY = Matrix{Float64}(undef, size(P))

    P = deepcopy(P)

    current_dim = dim(P)
    it = 0
    # Iterate until converged
    while true
        it += 1
        randomize!(X, P)
        randomize!(Y, P)
        LinearAlgebra.mul!(XY, X, Y)
        XY = _clamp_round!(XY, atol=atol)
        P = refine!(P, typeof(P)(XY))
        # Check if converged
        if current_dim == dim(P)
            break
        end
        current_dim = dim(P)
    end
    verbose && @info "desymmetization converged in $it iterations"
    return P
end
