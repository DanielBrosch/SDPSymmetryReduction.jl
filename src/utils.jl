function timed_print(io, t; msg=nothing)
    Base.time_print(io, t.time * 1.0e9, t.gcstats.allocd, t.gcstats.total_time, Base.gc_alloc_count(t.gcstats), t.lock_conflicts, t.compile_time * 1.0e9, t.recompile_time * 1.0e9, true; msg=msg)
end

"""
    clamptol(f::T; atol=Base.rtoldefault(T)) where T<:Number
Clamps numbers near zero to zero.
"""
function clamptol(f::Number; atol=Base.rtoldefault(real(typeof(f))))
    return ifelse(abs(f) < atol, zero(f), f)
end
clamptol!(f::Number; atol=Base.rtoldefault(real(typeof(f)))) = clamptol(f, atol=atol)
function clamptol!(a::AbstractArray)
    for (i, x) in pairs(a)
        a[i] = clamptol!(x)
    end
    # a .= clamptol!.(a)
    return a
end
function clamptol!(
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
