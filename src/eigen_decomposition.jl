struct EigenSpace{ED}
    parent::ED
    evecs_range::Base.UnitRange{Int}
end

Base.show(io::IO, ::MIME"text/plain", es::EigenSpace) =
    print(io, "Eigen-space of dimension ", length(range(es)), " with eigenvalue ", parent(es).values[first(range(es))])

Base.parent(es::EigenSpace) = es.parent
Base.range(es::EigenSpace) = es.evecs_range
dim(es::EigenSpace) = length(range(es))
vectors(es::EigenSpace) = @view vectors(parent(es))[:, range(es)]

struct EigenDecomposition{V,M}
    values::V
    vectors::M
    ptrs::Vector{Int}

    function EigenDecomposition(
        values::AbstractVector{T},
        vectors::AbstractMatrix{T};
        atol=1e-12 * length(values)
    ) where {T}
        ptrs = [1]
        for i in eachindex(values)
            if i == lastindex(values)
                push!(ptrs, i + 1)
                break
            end
            if !isapprox(values[i+1], values[i], atol=atol)
                # seems that the two values are actually different, we start new block!
                push!(ptrs, i + 1)
                # mn, mx = extrema(@view values[ptrs[end-1]:ptrs[end]-1])
                if abs(values[i+1] - values[i]) < eps(maximum(abs, @view values[i:i+1]))
                    @warn "Possibly numerically challenging example: no clear spectral gap "
                end
            end
        end
        return new{typeof(values),typeof(vectors)}(values, vectors, ptrs)
    end
end

function Base.show(io::IO, ::MIME"text/plain", es::EigenDecomposition)
    p = es.ptrs
    dims = [p[i+1] - p[i] for i in 1:length(p)-1]
    print(io, "Eigen-space decomposition ", dims)
end

Base.eltype(ed::EigenDecomposition) = eltype(ed.values)
Base.getindex(ed::EigenDecomposition, i::Integer) = EigenSpace(ed, ed.ptrs[i]:ed.ptrs[i+1]-1)
Base.length(ed::EigenDecomposition) = length(ed.ptrs) - 1
vectors(ed::EigenDecomposition) = ed.vectors

function Base.getindex(QᵀAQ::AbstractMatrix, es1::EigenSpace, es2::EigenSpace)
    @assert parent(es1) == parent(es2)
    @assert size(QᵀAQ) == size(parent(es1).vectors)
    return QᵀAQ[range(es1), range(es2)]
end

function Base.view(QᵀAQ::AbstractMatrix, es1::EigenSpace, es2::EigenSpace)
    @assert parent(es1) == parent(es2)
    @assert size(QᵀAQ) == size(parent(es1).vectors)
    return @view QᵀAQ[range(es1), range(es2)]
end

function block(A::AbstractMatrix, es1::EigenSpace, es2::EigenSpace)
    @assert parent(es1) == parent(es2)
    Qi = vectors(es1)
    Qj = vectors(es2)
    return Qi' * A * Qj
end

struct InvalidDecompositionField <: Exception
    requested
    found
end

Base.showerror(io::IO, e::InvalidDecompositionField) =
    print(
        io,
        """Decomposition over $(e.requested) was requested but eigenvalues of type $(e.found) were found.
        Consider calling `diagonalize` with $(e.found) as its first argument."""
    )

struct NumericalInconsistency <: Exception
    fn
    msg
end

Base.showerror(io::IO, e::NumericalInconsistency) =
    print(
        io,
        """Numerical inconsistency in $(e.fn):\n$(e.msg)"""
    )

function __isconsistent(K)
    Kpartition = [find_root!(K, i) for i in Base.OneTo(length(K))]
    roots = unique(Kpartition)
    return all(r == findfirst(==(r), Kpartition) for r in roots)
end

"""
    isomorphism_partition(ed::EigenDecomposition, A::AbstractMatrix[; atol])
Partition eigen-subspaces of `ed` into isomorphism classes based on a generic element `A`.

Computes partition `K` of Murota et al. as defined in **Algorithm 4.1**, Step 3.
"""
function isomorphism_partition(eigdec::EigenDecomposition, A::AbstractMatrix; atol=1e-12 * size(A, 1))
    Q = eigdec.vectors
    Q′AQ = Q' * A * Q

    neigspaces = length(eigdec)
    K = IntDisjointSets(neigspaces) # tracks the merging of eigenspaces
    for i in 1:neigspaces
        Ei = eigdec[i]
        for j in (i+1):neigspaces
            Ej = eigdec[j]
            if norm(@view Q′AQ[Ei, Ej], Inf) ≥ atol
                union!(K, i, j)
            end
        end
    end
    return K
end

"""
    eigen_decomposition(P::AbstractPartition, A::AbstractMatrix; atol=1e-12*size(A,1))
Find eigenspace decomposition of partition subspace `P` by inspecting a generic element thereof.

Returns `(ed::EigenDecomposition, K)` where `K` is a partition of eigen-spaces
of `ed` into isomorphism classes. If the computed `K` is suspected to be
inconsistent with `ed` (e.g. transitivity fails because of lack of precision)
`NumericalInconsistency` exception will be thrown.

Follows **Algorithm 4.1** of
> K. Murota et. al. A Numerical Algorithm for Block-Diagonal Decomposition of Matrix *-Algebras, Part I:
> Proposed Approach and Application to Semidefinite Programming
> _Japan Journal of Industrial and Applied Mathematics_, June 2010
> DOI: 10.1007/s13160-010-0006-9
"""
function eigen_decomposition(
    P::AbstractPartition,
    A::AbstractMatrix{T};
    atol=1e-12 * size(A, 1)
) where {T}
    # Step 1
    randomize!(A, P)

    # Step 2
    # eigenvalues and eigenvectors of A:
    F = eigen(A)
    Q, vals = try
        Q = convert(Matrix{T}, F.vectors)
        vals = convert(Vector{T}, F.values)
        Q, vals
    catch
        throw(InvalidDecompositionField(T, eltype(F)))
    end
    eigdec = EigenDecomposition(vals, Q; atol=atol)

    # Step 3
    # in case of Jordan algebras all Aₚ have disjoint support
    # this allows us to analyze Qᵢ′·Aₚ·Qⱼ all at once
    randomize!(A, P)

    # compute the equivalence relation K (eq. 4.2)
    K = isomorphism_partition(eigdec, A, atol=atol)

    if !__isconsistent(K)
        throw(
            NumericalInconsistency("eigen_decomposition",
                "the K-partition seems inconsistent with eigenspaces. Decrease `atol`, or simply try again."
            )
        )
    end

    return eigdec, K
end

# Murota et. al.
# The algorithm is described in section
# 4.3 Decomposition into irreducible components
# The last part follows Remark 4.1 therein.
function irreducible_decomposition(
    eigdec::EigenDecomposition,
    K::IntDisjointSets,
    P::AbstractPartition,
    A::AbstractMatrix=Matrix{eltype(eigdec)}(undef, size(P)),
)
    Kpartition = [find_root!(K, i) for i in Base.OneTo(length(K))]
    T = eltype(eigdec)
    roots = unique(Kpartition)

    P_hat = Vector{Matrix{T}}(undef, length(roots))
    A = randomize!(A, P)

    for (P_idx, i) in enumerate(roots)
        Ki = findall(==(i), Kpartition)
        @assert first(Ki) == i # below we make this assumption
        if length(Ki) == 1
            P_hat[P_idx] = vectors(eigdec[i])[:, 1:1]
            continue
        end
        # merge eigenspaces according to partition K
        QKi = reduce(hcat, [vectors(eigdec[i′]) for i′ in Ki])
        # Pi will be diagonal matrix diag(P₁, P₂, ..., P)  with each block of dimension
        dimEi = dim(eigdec[i]) # m_i in Murota;
        Pi = zeros(T, size(QKi, 2), size(QKi, 2))

        # The aim is to have P_hat = QKi*Pi with P_hat'*A*P_hat ≅ B⊗I

        # P₁ is easy
        idx = Base.OneTo(dimEi)
        Pi[idx, idx] .= I(dimEi)
        # for the rest we need to work
        Ei = eigdec[i]
        @views for (n, j) in enumerate(Ki[2:end])
            Ej = eigdec[j]
            blk = idx .+ n * dimEi
            P_blk = view(Pi, blk, blk)
            P_blk .= inv(block(A, Ei, Ej))
            # recover B_ij as the norm of the first row
            P_blk ./= norm(P_blk[1, :])
        end

        if dimEi == 1
            P_hat[P_idx] = QKi * Pi
        else
            # select the first column from each block
            first_columns = 1:dimEi:dimEi*length(Ki)
            P_hat[P_idx] = QKi * @view Pi[:, first_columns]
        end
    end

    return P_hat
end

