roundToZero(f::Number) = clamptol(f)
roundToZero!(a::AbstractArray) = clamptol_rec!(a)

function orthProject(A::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    return project_colspace(v, A)
end

part(M) = Partition(M)
coarsestPart(P, Q) = refine(P, Q)
rndPart(P) = randomize(Float64, P)
