function check_block_sizes(T::Type{<:Real}, Q, P::Partition, verbose::Bool)
    block_sizes = [size(q, 2) for q in Q]
    final_dim = sum(s -> (s + 1) * s ÷ 2, block_sizes)
    if final_dim ≠ P.n
        @error "Dimension mismatch over for $T:" (final_dim, block_sizes) expedcted_dim = P.n
        throw(DimensionMismatch("""Decomposition failed potentially due to
        * Rounding error (try different epsilons and/or try again) or
        * Algebra is not block-diagonalizable over the reals (retry with complex type).
        """))
    end
end

function check_block_sizes(T::Type{<:Complex}, Q, P::Partition, verbose::Bool)
    block_sizes = [size(q, 2) for q in Q]
    final_dim = sum(s -> s^2, block_sizes)
    if final_dim ≠ P.n
        @error "Dimension mismatch over for $T:" (final_dim, block_sizes) expedcted_dim = P.n
        throw(DimensionMismatch("""Decomposition failed potentially due to
        * Rounding error (try different epsilons and/or try again) or
        * Unknow reason, please consider submitting an issue.
         """))
    end
end

function diagonalize(::Type{T}, P::Partition, verbose=true; epsilon=Base.rtoldefault(real(T))) where {T<:Number}
    A = Matrix{T}(undef, size(P.P))

    verbose && @info "Determining eigen-decomposition over $T..."
    t = @timed eigdec, K = eigen_decomposition(P, A; atol=atol)
    verbose && @info sprint(timed_print, t)

    verbose && @info "Determining the algebra-isomorphism..."
    t = @timed Q_hat = irreducible_decomposition(eigdec, K, P, A)
    verbose && @info sprint(timed_print, t)

    return Q_hat
end

function _constraints(P::Partition)
    cnstrs = [UInt32[] for _ in 1:P.n]
    for li in eachindex(IndexLinear(), P.P)
        v = P.P[li]
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

function basis_image(Q::AbstractVector{<:AbstractMatrix}, P::Partition)
    T = eltype(first(Q))
    basis_img = Vector{Vector{Matrix{T}}}(undef, P.n)

    tmp = let tmp = Vector{Matrix{T}}(undef, length(Q))
        for (i, q) in zip(eachindex(tmp), Q)
            tmp[i] = Matrix{T}(undef, size(P.P, 1), size(q, 2))
        end
        tmp
    end

    C = _constraints(P)
    M = spzeros(T, Int, size(P.P))

    for i in eachindex(basis_img, C)
        M = sparse_constraint!(M, C[i], 1.0)
        basis_img[i] = [conjugate(M, Qi, t) for (Qi, t) in zip(Q, tmp)]
        clamptol_rec!(basis_img[i])
    end

    return basis_img
end

function basis_image_thr(Q::AbstractVector{<:AbstractMatrix}, P::Partition)
    T = eltype(first(Q))
    basis_img = Vector{Vector{Matrix{T}}}(undef, P.n)
    C = _constraints(P)
    Threads.@threads for i in eachindex(basis_img, C)
        M = spzeros(Float64, Int, size(P.P))
        M[C[i]] .= 1
        basis_img[i] = [conjugate(M, Qi) for Qi in Q]
        clamptol_rec!(basis_img[i])
    end
    return basis_img
end
