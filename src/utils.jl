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

function clamptol!(
    a::AbstractArray{T};
    atol=Base.rtoldefault(real(T))
) where {T<:Number}
    for (i, x) in pairs(a)
        a[i] = clamptol(x; atol=atol)
    end
    return a
end
function clamptol!(
    a::SparseMatrixCSC{T};
    atol=Base.rtoldefault(real(typeof(f)))
) where {T<:Number}
    return droptol!(a, atol)
end

function _clamp_round!(
    A::AbstractArray;
    atol=Base.rtoldefault(real(eltype(A))),
    sigdigits=floor(Int, -log10(atol))
)
    @inbounds for (i, a) in pairs(A)
        A[i] = ifelse(
            abs(a) < atol,
            zero(a),
            unsafe_round(a, scale=10^sigdigits)
        )
    end
    return A
end

function unsafe_round(f::AbstractFloat; scale)
    (x, n) = frexp(f)
    y = unsafe_trunc(Int, scale * x) / scale
    return ldexp(y, n)
end

"""
    project_colspace(v::AbstractVector{T}, A::AbstractMatrix{T}) where T
Project v orthogonally to the column space of A.
"""
function project_colspace(v::AbstractVector, A::AbstractMatrix; Afact=qr(A))
    return project_colspace!(similar(v), v, A; Afact=Afact)
end
function project_colspace!(res, v::AbstractVector, A; Afact=qr(A))
    # return A * (Afact \ v)
    mul!(res, A, Afact \ v)
    return res
end

project_colspace(v::SparseVector, A::AbstractMatrix; Afact=qr(A)) =
    project_colspace(Vector(v), A; Afact=Afact)

function _symmetrize!(v::AbstractVector{T}, n::Integer) where {T}
    @assert length(v) == n^2
    M = reshape(v, n, n)
    @inbounds for i in axes(M, 2)
        for j in i:size(M, 1)
            t = (M[i, j] + M[j, i]) / 2
            M[i, j] = M[j, i] = t
        end
    end
    return v
end
