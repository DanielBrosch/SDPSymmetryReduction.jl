roundToZero(f::Number) = clamptol(f)
roundToZero!(a::AbstractArray) = clamptol_rec!(a)

function orthProject(A::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    return project_colspace(v, A)
end

part(M) = Partition(M)
coarsestPart(P, Q) = refine(P, Q)
rndPart(P) = randomize(Float64, P)

roundMat(M) = M .= clamptol.(round.(M, sigdigits=5))

function projectAndRound(M, A; round=true)
    v = vec(M)
    v .-= orthProject(A, v)
    if round
        roundMat(v)
    end
    return Float64.(reshape(v, size(M)...))
end

admPartSubspace(C, A, b, verbose=false) = admissible_subspace(C, A, b, verbose)
