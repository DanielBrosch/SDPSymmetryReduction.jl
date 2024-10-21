roundToZero(f::Number) = clamptol(f)
roundToZero!(a::AbstractArray) = clamptol_rec!(a)

function orthProject(A::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    return project_colspace(v, A)
end
