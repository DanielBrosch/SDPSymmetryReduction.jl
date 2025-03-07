function check_block_sizes(T::Type{<:Real}, Q, P::AbstractPartition, verbose::Bool)
    block_sizes = [size(q, 2) for q in Q]
    final_dim = sum(s -> (s + 1) * s ÷ 2, block_sizes)
    if final_dim ≠ dim(P)
        @error "Dimension mismatch over for $T:" (final_dim, block_sizes) expedcted_dim = dim(P)
        throw(DimensionMismatch("""Decomposition failed potentially due to
        * Rounding error (try different epsilons and/or try again) or
        * Algebra is not block-diagonalizable over the reals (retry with complex type).
        """))
    end
end

function check_block_sizes(T::Type{<:Complex}, Q, P::AbstractPartition, verbose::Bool)
    block_sizes = [size(q, 2) for q in Q]
    final_dim = sum(s -> s^2, block_sizes)
    if final_dim ≠ dim(P)
        @error "Dimension mismatch over for $T:" (final_dim, block_sizes) expedcted_dim = dim(P)
        throw(DimensionMismatch("""Decomposition failed potentially due to
        * Rounding error (try different epsilons and/or try again) or
        * Unknown reason, please consider submitting an issue.
         """))
    end
end

function diagonalize(::Type{T}, P::AbstractPartition; verbose=false, atol=1e-12 * size(P, 1)) where {T<:Number}
    if T <: Complex
        P = desymmetrize(P; verbose=verbose)
    end
    A = Matrix{T}(undef, size(P))

    verbose && @info "Determining eigen-decomposition over $T..."
    t = @timed eigdec, K = eigen_decomposition(P, A; atol=atol)
    verbose && @info sprint(timed_print, t)

    verbose && @info "Determining the algebra-isomorphism..."
    t = @timed Q_hat = irreducible_decomposition(eigdec, K, P, A)
    verbose && @info sprint(timed_print, t)

    return clamptol!.(Q_hat, atol=atol)
end

function _constraints(P::Partition)
    cnstrs = [UInt32[] for _ in 1:dim(P)]
    for li in eachindex(IndexLinear(), P.matrix)
        v = P.matrix[li]
        iszero(v) && continue
        push!(cnstrs[v], li)
    end
    return cnstrs
end

function sparse_constraint!(M::SparseMatrixCSC, indices, val=one(eltype(M)))
    M .= 0
    dropzeros!(M)
    M[indices] .= val
    return M
end

function conjugate(M, Q, tmp=Matrix{eltype(Q)}(undef, size(M, 1), size(Q, 2)))
    mul!(tmp, M, Q)
    return Q' * tmp
end

function basis_image(
    Q::AbstractVector{<:AbstractMatrix},
    P::AbstractPartition;
    atol=1e-12 * size(P, 1),
)
    T = eltype(first(Q))
    basis_img = Vector{Vector{Matrix{T}}}(undef, dim(P))

    tmp = let tmp = Vector{Matrix{T}}(undef, length(Q))
        for (i, q) in zip(eachindex(tmp), Q)
            tmp[i] = Matrix{T}(undef, size(P, 1), size(q, 2))
        end
        tmp
    end

    C = _constraints(P)
    M = spzeros(T, Int, size(P))

    for i in eachindex(basis_img, C)
        Pi = sparse_constraint!(M, C[i], 1.0)
        basis_img[i] = [conjugate(Pi, Qi, t) for (Qi, t) in zip(Q, tmp)]
        clamptol!.(basis_img[i], atol=atol)
    end

    return basis_img
end

function basis_image_thr(
    Q::AbstractVector{<:AbstractMatrix},
    P::AbstractPartition;
    atol=1e-12 * size(P, 1),
)
    T = eltype(first(Q))
    basis_img = Vector{Vector{Matrix{T}}}(undef, dim(P))
    C = _constraints(P)
    Threads.@threads for i in eachindex(basis_img, C)
        M = spzeros(Float64, Int, size(P))
        M[C[i]] .= 1
        basis_img[i] = [conjugate(M, Qi) for Qi in Q]
        clamptol!.(basis_img[i], atol=atol)
    end
    return basis_img
end
