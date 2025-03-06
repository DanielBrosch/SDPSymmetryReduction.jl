import JuMP, MutableArithmetics

abstract type AbstractCAb{T} end

Base.iterate(L::AbstractCAb) = (L.C, Val(:A))
Base.iterate(L::AbstractCAb, ::Val{:A}) = (L.A, Val(:b))
Base.iterate(L::AbstractCAb, ::Val{:b}) = (L.b, nothing)
Base.iterate(::AbstractCAb, ::Nothing) = nothing

struct Lovászϑ′{T,Ct<:AbstractVector{T},At<:AbstractMatrix{T},bt<:AbstractVector{T}} <: AbstractCAb{T}
    C::Ct
    A::At
    b::bt
end

function Lovászϑ′_ER_graph(q::Integer)
    PG2q = vcat([[0, 0, 1]],
        [[0, 1, b] for b = 0:q-1],
        [[1, a, b] for a = 0:q-1 for b = 0:q-1])
    Adj = [x' * y % q == 0 && x != y for x in PG2q, y in PG2q]

    N = length(PG2q)
    C = ones(N^2)
    A = vcat(vec(Adj)', vec(Matrix{Float64}(I, N, N))')
    b = [0.0, 1.0]
    return Lovászϑ′(C, A, b)
end

function opt_model(P, Q_hat, CAb::Lovászϑ′)
    d = SDPSymmetryReduction.dim(P)
    N = prod(size(P))
    cnstrs = SDPSymmetryReduction._constraints(P)
    PMat = hcat([(v = falses(N); v[c] .= true; v) for c in cnstrs]...)
    # PMat = hcat([sparse(vec(P.matrix .== i)) for i = 1:d]...)
    A = CAb.A * PMat
    B = CAb.b
    C = CAb.C' * PMat

    m = JuMP.Model()
    # Initialize variables corresponding parts of the partition P
    # >= 0 because the original SDP-matrices are entry-wise nonnegative
    x = JuMP.@variable(m, x[1:d] >= 0)
    JuMP.@objective(m, Max, C * x)
    JuMP.@constraint(m, A * x .== B)

    blks = SDPSymmetryReduction.basis_image(Q_hat, P)
    psd = MutableArithmetics.@rewrite sum(
        blks[i] .* x[i] for i = 1:d
    )
    for p in psd
        JuMP.@constraint(m, p in JuMP.PSDCone())
    end

    return m
end

struct QuadraticAssignment{T,Ct<:AbstractVector{T},At<:AbstractMatrix{T},bt<:AbstractVector{T}} <: AbstractCAb{T}
    C::Ct
    A::At
    b::bt
end

function __qap_Ab(n)
    In = sparse(I, n, n)
    Jn = trues(n, n)

    A = spzeros(2n + 1, n^4)
    b = zeros(2n + 1)
    currentRow = 1

    for j = 1:n
        Ejj = spzeros(n, n)
        Ejj[j, j] = 1.0
        A[currentRow, :] = vec(kron(In, Ejj))
        b[currentRow] = 1
        currentRow += 1
        # Last constraint is linearly dependent on others
        if (j < n)
            A[currentRow, :] = vec(kron(Ejj, In))
            b[currentRow] = 1
            currentRow += 1
        end
    end

    A[currentRow, :] = vec(kron(In, Jn - In) + kron(Jn - In, In))
    b[currentRow] = 0
    currentRow += 1
    fill!(@view(A[currentRow, :]), 1)
    b[currentRow] = n^2

    A, b
end

function QuadraticAssignment(flowA::AbstractMatrix{T}, flowB::AbstractMatrix{T}) where {T}
    n, m = LinearAlgebra.checksquare(flowA, flowB)
    @assert n == m
    A, b = __qap_Ab(n)

    C = kron(flowA, flowB)
    if !issymmetric(C)
        C .= (C .+ C') ./ 2
    end

    QuadraticAssignment(sparse(vec(C)), A, b)
end

function opt_model(P, Q_hat, CAb::QuadraticAssignment)
    d = SDPSymmetryReduction.dim(P)
    N = length(CAb.C)
    n = isqrt(isqrt(length(CAb.C)))
    @assert n^4 == N

    cnstrs = SDPSymmetryReduction._constraints(P)
    PMat = hcat([(v = falses(n^4); v[c] .= true; v) for c in cnstrs]...)

    C = CAb.C' * PMat
    A = CAb.A * PMat
    b = CAb.b

    m = JuMP.Model()

    # Initialize variables corresponding parts of the partition P
    # >= 0 because the original SDP-matrices are entry-wise nonnegative
    x = JuMP.@variable(m, x[1:d] >= 0)
    JuMP.@objective(m, Min, C * x)
    JuMP.@constraint(m, A * x == b)
    blks = SDPSymmetryReduction.basis_image(Q_hat, P)
    psd = MutableArithmetics.@rewrite(
        sum(x[i] * blks[i] for i = 1:d)
    )

    for p in psd
        JuMP.@constraint(m, p in JuMP.PSDCone())
    end

    return m
end
