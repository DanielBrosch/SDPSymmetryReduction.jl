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

"""
    admissible_subspace(C, A, b[, verbose; rtol])
Returns the optimal admissible partition subspace for the SD problem

``\\inf\\{C^Tx, Ax = b, \\mathrm{Mat}(x) \\succcurlyeq 0\\}.``

The problem can be restricted to the subspace without changing its optimal value.

The partition is found by a Jordan-reduction algorithm saturating with __random__
squares. With probability 1 the returned subspace is a Jordan algebra
(i.e. closed under linear combinations and squaring).
See Section 5.2 of Permenter thesis.

# Output
* A `Partition` subspace `P`.
"""
function admissible_subspace(
    C::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    verbose::Bool=false;
    rtol=Base.rtoldefault(real(T))
) where {T<:AbstractFloat}

    sigdigits = ceil(Int, -log10(rtol))
    n = isqrt(length(C))
    @assert n^2 == length(C)
    # temporary values for re-use
    X = Matrix{Float64}(undef, n, n)
    X² = Matrix{Float64}(undef, n, n)

    projLᵖ = let A′ = A', A′qr = qr(A′)
        v -> project_colspace(v, A′, Afact=A′qr)
    end

    # notation according to Brosch
    CL = let c = C
        c = c - projLᵖ(c) # we own `c` from now on
        c .= clamptol.(round.(c, sigdigits=sigdigits))
        c = _symmetrize!(c, n)
        reshape(c, n, n)
    end

    X₀Lᵖ = let X = A \ b # we own `X₀`
        X = _symmetrize!(X, n)
        X = projLᵖ(X)
        X .= clamptol.(round.(X, sigdigits=sigdigits))
        reshape(X, n, n)
    end

    # Initialize S as the span of the initial two elements
    S = Partition(CL)
    S = refine(S, Partition(X₀Lᵖ))

    maximal_dimension = (n^2 + n) ÷ 2
    current_dimension = S.n
    verbose && @info "Starting the reduction. Dimensions:" maximal = maximal_dimension initial = current_dimension

    it = 0
    # Iterate until converged
    while current_dimension < maximal_dimension
        it += 1
        verbose && @info "Iteration $it, Current dimension: $current_dimension"

        # Add a random projection to S
        X = randomize!(X, S)
        let x = vec(X) # x in here shares memory with X!
            x .-= projLᵖ(x)
            x .= clamptol.(round.(x, sigdigits=sigdigits))
        end
        S = refine(S, Partition(X))

        if current_dimension != S.n
            X = randomize!(X, S)
        end

        # Add random square
        X² = mul!(X², X, X)
        X² .= clamptol.(round.(X², sigdigits=sigdigits))

        S = refine(S, Partition(X²))

        # with probability 1 a random square does not refine S
        # only when S is already closed under taking squares,
        # i.e. it a Jordan subalgebra, at least for special ones.
        # See Theorem 5.2.3 in Permenter thesis
        if current_dimension == S.n # converged
            break
        end

        current_dimension = S.n
    end

    verbose &&
        @info "Minimal admissible subspace converged in $it iterations. Dimensions:" initial = maximal_dimension final = S.n
    return S
end
