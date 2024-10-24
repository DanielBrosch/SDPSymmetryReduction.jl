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

function Base.getindex(QᵀAQ::AbstractArray, es1::EigenSpace, es2::EigenSpace)
    @assert parent(es1) == parent(es2)
    @assert size(QᵀAQ) == size(parent(es1).vectors)
    return @view QᵀAQ[range(es1), range(es2)]
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

# Murota et. al., Algorithm 4.1
function eigen_decomposition(
    P::Partition,
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
    ED = EigenDecomposition(vals, Q; atol=atol)

    # Step 3
    # in case of Jordan algebras all Aₚ have disjoint support
    # this allows us to analyze Qᵢ′·Aₚ·Qⱼ all at once
    randomize!(A, P)
    Q′AQ = Q' * A * Q

    # compute the equivalence relation K (eq. 4.2)
    neigspaces = length(ED)
    K = IntDisjointSets(neigspaces) # tracks the merging of eigenspaces
    for i in 1:neigspaces
        for j in (i+1):neigspaces
            in_same_set(K, i, j) && continue
            Ei, Ej = ED[i], ED[j]
            dim(Ei) != dim(Ej) && continue
            # TODO: numerical condition for non-zero block
            # if the maximal element not small relative to 1.0...
            if any(x -> abs(x) ≥ atol, Q′AQ[Ei, Ej])
                # Wedderburn-Artin decomposition:
                # since Q′AQ[esi, esj] : esj → esi is non-zero, it must be
                # an isomorphism, so we merge
                union!(K, i, j)
            end
        end
    end

    return ED, K
end

# Murota et. al.
# The algorithm is described in section
# 4.3 Decomposition into irreducible components
# The last part follows Remark 4.1 therein.
function irreducible_decomposition(
    eigdec::EigenDecomposition,
    K::IntDisjointSets,
    P::Partition,
    A::AbstractMatrix=Matrix{eltype(eigdec)}(undef, size(P.P)),
)
    Kpartition = [find_root!(K, i) for i in Base.OneTo(length(K))]
    T = eltype(eigdec)
    roots = unique(Kpartition)

    P_hat = Vector{Matrix{T}}(undef, length(roots))
    A = randomize!(A, P)

    for P_idx in eachindex(roots, P_hat)
        root = roots[P_idx]
        Ki = findall(==(root), Kpartition)
        n_merged_blks = length(Ki) # how many eigen-blocks are we merging together?
        # merge eigenspaces according to partition K
        QKi = reduce(hcat, [vectors(eigdec[i]) for i in Ki])

        QKᵢ′AQKᵢ = QKi' * A * QKi

        # Pi will be diagonal matrix diag(P₁, P₂, ..., P)
        Pi = zeros(T, size(QKᵢ′AQKᵢ))

        # each block is of this dimension
        eigval_mult = dim(eigdec[root]) # m_i in Murota;
        @assert eigval_mult * n_merged_blks == size(QKᵢ′AQKᵢ, 1)

        idx = Base.OneTo(eigval_mult)
        # The aim is to have QKi*P ≅ B⊗I
        Pi[idx, idx] .= I(eigval_mult)
        @views for j in 2:n_merged_blks
            shift = eigval_mult * (j - 1)
            blk = shift .+ idx
            Pi[blk, blk] .= inv(QKᵢ′AQKᵢ[idx, blk]) # so that we get identity
            Pi[blk, blk] ./= norm(Pi[shift+1, blk]) # recover b_ij
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

