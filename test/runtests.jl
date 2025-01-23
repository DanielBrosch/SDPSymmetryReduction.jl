using Test
using LinearAlgebra
using SparseArrays
using SDPSymmetryReduction
import CSDP

@testset "SDPSymmetryReduction.jl" begin

    dim = SDPSymmetryReduction.dim

    @test iszero(SDPSymmetryReduction.roundToZero(1e-10))

    M = rand(1:10, 10, 10)
    M[1] = 0
    @test dim(SDPSymmetryReduction.part(M)) == length(unique(M)) - 1
    @test dim(SDPSymmetryReduction.part(Float64.(M))) == length(unique(M)) - 1

    M = rand(1:10, 10, 10)
    @test dim(SDPSymmetryReduction.part(M)) == length(unique(M))
    @test dim(SDPSymmetryReduction.part(Float64.(M))) == length(unique(M))

    P1 = Partition([1 2 2; 2 3 3; 2 3 3])
    P2 = Partition([1 1 2; 1 1 2; 1 1 3])
    P3 = Partition([1 2 4; 2 3 5; 2 3 6])
    @test SDPSymmetryReduction.coarsestPart(P1, P2) == P3

    @test SDPSymmetryReduction.part(SDPSymmetryReduction.rndPart(P1)) == P1

    M = rand(10, 10)
    @test SDPSymmetryReduction.roundMat(M) ≈ M atol = 1e-4

    A = rand(9, 3)
    M = rand(3, 3)
    @test maximum(abs.(A \ vec(SDPSymmetryReduction.projectAndRound(M, A; round=false)))) ≈ 0.0 atol = 1e-10

    T = M - SDPSymmetryReduction.projectAndRound(M, A; round=false)
    @test all(isapprox.(A * (A \ vec(T)) - vec(T), 0, atol=1e-8))

    @testset "unsymmetization and complex" begin
        @test SDPSymmetryReduction.unSymmetrize(P1) == Partition(4, [1 3 3; 2 4 4; 2 4 4])

        # complex block diagonalization tests
        P = Partition(3, [1 2 3 2; 2 1 2 3; 3 2 1 2; 2 3 2 1])

        @test issymmetric(SDPSymmetryReduction.randomize(P))

        @test blockDiagonalize(P, true; complex=true).blkSizes == [1, 1, 1]

        # cyclic group of order 3:
        C₃ = [
            1 2 3
            2 3 1
            3 1 2
        ]
        C₃ = reverse(C₃, dims=1)
        P₃ = Partition(C₃)
        @test_throws SDPSymmetryReduction.InvalidDecompositionField blockDiagonalize(P₃)
        @test blockDiagonalize(P₃, complex=true).blkSizes == [1, 1, 1]
    end

    include("sd_problems.jl")
    include("lovasz.jl")
    include("qap.jl")

    include("partitions_set.jl")
end
