module PartitionInPlace

import ..SDPSymmetryReduction as SR
using Test

"""
    Partition

A partition subspace stored internally as matrix of integers from `0` to `dim(P)`.
"""
struct Partition{T<:Integer} <: SR.AbstractPartition
    nparts::Ref{Int} # Number of parts
    matrix::Matrix{T} # Matrix with entries 1,...,n
    Partition{T}(nparts, matrix) where {T} = new{T}(Ref(Int(nparts)), matrix)
end

Partition(args...) = Partition{UInt16}(args...)

SR.dim(p::Partition) = p.nparts[]
Base.size(p::Partition, args...) = size(p.matrix, args...)


"""
    Partition(M::AbstractMatrix)

Create a partition from the unique entries of `M`.
"""
function Partition{T}(M::AbstractMatrix) where {T}
    l = 0
    d = Dict(zero(eltype(M)) => T(l))
    res = zeros(T, size(M))
    for (v, idx) in zip(M, eachindex(res))
        k = get!(d, v, T(l + 1))
        l = ifelse(k == l + 1, k, l)
        res[idx] = k
    end

    return Partition{T}(l, res)
end

function Partition{T}(M::AbstractMatrix{<:Integer}) where {T}
    M_vals = unique(M)
    @assert 0 ≤ first(M_vals)
    vals = zeros(Int, maximum(M_vals) + 1) # to accomodate for 0 if it exists
    dim = 0
    for v in M_vals
        iszero(v) && continue # to preserve 0
        dim += 1
        vals[v+1] = dim
    end
    res = zeros(T, size(M))
    for (v, idx) in zip(M, eachindex(res))
        res[idx] = vals[v+1]
    end
    return Partition{T}(dim, res)
end

function sort_unique!(M::AbstractMatrix{<:Integer})
    M_vals = unique(M)
    @assert 0 ≤ first(M_vals)
    # vals = Dict{Int, Int}(1 => 0) # doesn't seem worth it, even with gaps
    vals = zeros(Int, maximum(M_vals) + 1) # to accomodate for 0 if it exists
    dim = 0
    for v in M_vals
        iszero(v) && continue # to preserve 0
        dim += 1
        vals[v+1] = dim
    end
    for (v, idx) in zip(M, eachindex(M))
        M[idx] = vals[v+1]
    end
    return M, dim
end

function SR.refine!(P1::Partition{T}, P2::Partition{S}) where {T,S}
    P1.matrix .+= P2.matrix .* (SR.dim(P1) + 1)
    _, d = sort_unique!(P1.matrix)
    P1.nparts[] = d
    return P1
    # return Partition{T}(d, P1.matrix)
end

function SR._constraints(P::Partition)
    cnstrs = [UInt32[] for _ in 1:SR.dim(P)]
    for li in eachindex(IndexLinear(), P.matrix)
        v = P.matrix[li]
        iszero(v) && continue
        push!(cnstrs[v], li)
    end
    return cnstrs
end

# function refine!(P1::Partition{T}, P2::Partition{S}) where {T,S}
#     P2.matrix .= P1.matrix .+ P2.matrix .* (dim(P1) + 1)
#     return Partition{T}(P2.matrix)
# end

function Base.fill!(M::AbstractMatrix{<:Real}, P::Partition; values::AbstractVector)
    for idx in eachindex(P.matrix, M)
        M[idx] = values[P.matrix[idx]+1]
    end
    return M
end

end

import .PartitionInPlace as PIP

@testset "Lovász ϑ′ for Erdös-Renyi graphs" begin
    dim = SDPSymmetryReduction.dim
    @testset "PGL(2,q=3)" begin
        CAb = Lovászϑ′_ER_graph(3)
        P = SDPSymmetryReduction.admissible_subspace(
            PIP.Partition,
            CAb...
        )
        @test SDPSymmetryReduction.dim(P) == 12
        Q_hat = SDPSymmetryReduction.diagonalize(Float64, P)
        @test sort(size.(Q_hat, 2)) == [2, 2, 3]

        model = opt_model(P, Q_hat, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 5.0 rtol = 1e-7
    end

    @testset "qap problem: esc16j" begin
        esc16_file = joinpath(@__DIR__, "qapdata", "esc16j.dat")

        A, B = read_qapdata(esc16_file, Float64)
        CAb = QuadraticAssignment(A, B)
        P = SDPSymmetryReduction.admissible_subspace(
            PIP.Partition,
            CAb...,
        )
        @test SDPSymmetryReduction.dim(P) == 150

        Q_hat = SDPSymmetryReduction.diagonalize(Float64, P)
        @test sort(size.(Q_hat, 2)) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7]

        model = opt_model(P, Q_hat, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 7.7942186 rtol = 1e-7
    end
end
