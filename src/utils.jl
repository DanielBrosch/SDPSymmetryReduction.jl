"""
    clamptol(f::T; atol=Base.rtoldefault(T)) where T<:Number

Clamps numbers near zero to zero.
"""
function clamptol(f::Number; atol=Base.rtoldefault(real(typeof(f))))
    return ifelse(abs(f) < atol, zero(f), f)
end

clamptol_rec!(f::Number; atol=Base.rtoldefault(real(typeof(f)))) = clamptol(f, atol=atol)
function clamptol_rec!(a::AbstractArray)
    a .= clamptol_rec!.(a)
    return a
end

function clamptol_rec!(
    a::SparseMatrixCSC{T};
    atol=Base.rtoldefault(real(typeof(f)))
) where {T<:Number}
    return droptol!(a, atol)
end

"""
    project_colspace(v::AbstractVector{T}, A::AbstractMatrix{T}) where T
Project v orthogonally to the column space of A.
"""
function project_colspace(v::AbstractVector, A::AbstractMatrix; Afact=qr(A))
    return A * (Afact \ v)
end

project_colspace(v::SparseVector, A::AbstractMatrix; Afact=qr(A)) =
    project_colspace(Vector(v), A; Afact=Afact)

function _symmetrize!(v::AbstractVector{T}, n::Integer) where {T}
    @assert length(v) == n^2
    M = reshape(v, n, n)
    M .+= M'
    M ./= T(2)
    return v
end
