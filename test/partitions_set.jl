module PartitionSet

import ..SDPSymmetryReduction as SR
using Test

struct Partition{S<:AbstractSet} <: SR.AbstractPartition
    size::Tuple{Int,Int}
    zero_set::S
    sets::Vector{S}
end

SR.dim(p::Partition) = length(p.sets)
Base.size(p::Partition) = p.size
Base.size(p::Partition, i::Integer) = p.size[i]

function Partition{S}(M::AbstractMatrix{<:Number}) where {S<:AbstractSet}
    Base.require_one_based_indexing(M)

    l = 0
    d = Dict(zero(eltype(M)) => l)
    sets = [S()]
    for idx in eachindex(IndexLinear(), M)
        v = M[idx]
        k = get!(d, v, l + 1)
        if k == l + 1
            push!(sets, S(idx))
            l += 1
        else
            push!(sets[d[v]+1], idx)
        end
    end
    zero_set = popfirst!(sets)
    return Partition(size(M), zero_set, sets)
end

function SR.refine!(P₁::Partition{S}, P₂::Partition) where {S}
    @assert size(P₁) == size(P₂)
    splits = Dict{Int,S}()
    if P₁.zero_set ≠ P₂.zero_set
        push!(P₂.sets, setdiff(P₁.zero_set, P₂.zero_set))
        intersect!(P₁.zero_set, P₂.zero_set)
        setdiff!(P₂.zero_set, P₁.zero_set)
        push!(P₂.sets, P₂.zero_set)
    end
    for S₂ in P₂.sets
        splits = empty!(splits)
        for s₂ in S₂
            idx = something(findfirst(S -> s₂ in S, P₁.sets), 0)
            P₁_set = iszero(idx) ? P₁.zero_set : P₁.sets[idx]
            delete!(P₁_set, s₂)
            if haskey(splits, idx)
                push!(splits[idx], s₂)
            else
                splits[idx] = S(s₂)
                sizehint!(splits[idx], length(P₁_set))
            end
        end
        for (i, S₁) in pairs(P₁.sets)
            if isempty(S₁)
                union!(S₁, splits[i])
                delete!(splits, i)
            end
        end
        append!(P₁.sets, values(splits))
    end
    return P₁
end

function Base.fill!(M::AbstractMatrix{<:Real}, P::Partition; values::AbstractVector)
    @assert length(values) == length(P.sets)
    @inbounds for i in P.zero_set
        M[i] = zero(eltype(M))
    end
    @inbounds for (S, v) in zip(P.sets, values)
        for i in S
            M[i] = v
        end
    end
    return M
end

function __matrix(P::Partition) # convienience
    M = zeros(Int, size(P))
    for (i, s) in enumerate(P.sets)
        for lidx in s
            M[lidx] = i
        end
    end
    return M
end

SR._constraints(P::Partition) = collect.(P.sets)

end

import .PartitionSet as PS

@testset "Lovász ϑ′ for Erdös-Renyi graphs" begin
    dim = SDPSymmetryReduction.dim
    @testset "PGL(2,q=3)" begin
        CAb = Lovászϑ′_ER_graph(3)
        P = SDPSymmetryReduction.admissible_subspace(
            PS.Partition{BitSet},
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

    @static if !Sys.IsWindows()
        @testset "qap problem: esc16j" begin
            esc16_file = joinpath(@__DIR__, "qapdata", "esc16j.dat")

            A, B = read_qapdata(esc16_file, Float64)
            CAb = QuadraticAssignment(A, B)
            P = SDPSymmetryReduction.admissible_subspace(
                PS.Partition{BitSet},
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
end
