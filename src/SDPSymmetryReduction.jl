module SDPSymmetryReduction

using LinearAlgebra
using Random
using SparseArrays

export Partition, admPartSubspace, blockDiagonalize
include("utils.jl")
include("compat.jl")
include("partitions.jl")


"""
    unSymmetrize(P::Partition)

WL algorithm to "un-symmetrize" the Jordan algebra corresponding to `P`.
"""
function unSymmetrize(P::Partition)
    P = deepcopy(P)
    dim = P.n
    it = 0
    # Iterate until converged
    while true
        it += 1
        randomizedP1 = rndPart(P)
        randomizedP2 = rndPart(P)
        P2 = randomizedP1 * randomizedP2
        P2 = roundMat(P2)
        P = coarsestPart(P, part(P2))
        # Check if converged
        if dim == P.n
            break
        end
        dim = P.n
    end
    return P
end
unSymmetrize(q::Matrix) = unSymmetrize(Partition(q))
export unSymmetrize

# modifies r and A in place according to P
function getRandomMatrix!(r, A, P; complex = false)
    rand!(r)
    r[1] = 0 # write it explicitly for clarity
    @inbounds for i in eachindex(P.P)
        A[i] = r[P.P[i]+1]
    end
    if complex
        A .+= A'
    end
    return A
end

"""
    blockDiagonalize(P::Partition, verbose = true; epsilon = 1e-8, complex = false)

Determines a block-diagonalization of a (Jordan)-algebra given by a partition `P` using a randomized algorithm. `blockDiagonalize(P)` returns a real block-diagonalization `blkd`, if it exists, otherwise `nothing`.

`blockDiagonalize(P; complex = true)` returns the same, but with complex valued matrices, and should be used if no real block-diagonalization was found. To use the complex matrices practically, remember that a Hermitian matrix `A` is positive semidefinite iff `[real(A) -imag(A); imag(A) real(A)]` is positive semidefinite.

## Output

* `blkd.blkSizes` is an integer array of the sizes of the blocks.
* `blkd.blks` is an array of length `P.n` containing arrays of (real/complex) matrices of sizes `blkd.blkSizes`. I.e. `blkd.blks[i]` is the image of the basis element `P.P .== i`.
"""
function blockDiagonalize(::Type{T}, P::Partition, verbose = true; epsilon = Base.rtoldefault(real(T))) where {T <: Number}
    complex = T <: Complex
    if complex
        P = unSymmetrize(P)
    end

    verbose && println("Determining block sizes...")

    @time begin

    r = Vector{T}(undef, P.n + 1)
    A = Matrix{T}(undef, size(P.P)) # used for getRandomMatrix!
    B = Matrix{T}(undef, size(P.P)) # used for V' * A * V


    getRandomMatrix!(r, A, P; complex)
    F = eigen(A)
    Q = F.vectors

    # split by eigenvalues
    roundedEV = round.(F.values; digits = 10)
    uniqueEV = unique(roundedEV)
    indEV = [[i for i in 1:length(roundedEV) if roundedEV[i] == u] for u in uniqueEV]
    countEV = length.(indEV)
    csEV = vcat([0], cumsum(countEV))
    csEV .+= 1

    K = collect(1:length(uniqueEV))
    getRandomMatrix!(r, A, P; complex)
    mul!(B, A, Q)
    mul!(A, Q', B)
    for i in 1:length(uniqueEV), j in i+1:length(uniqueEV)
        if K[i] != K[j] && countEV[i] == countEV[j]
                A_ij = A[csEV[i]:csEV[i+1]-1, csEV[j]:csEV[j+1]-1]
                if any(x -> abs(x) ≥ epsilon, A_ij)
                K[K.==K[j]] .= K[i]
            end
        end
    end
    ind_uniqueK = unique(i -> K[i], eachindex(K))
    uniqueKs = K[ind_uniqueK]
    blockSizes = [sum(1 for elK in K if elK == Ki) for Ki in uniqueKs]

    end
    verbose && println("Block sizes are $(sort(blockSizes))")

    if !complex && sum(x -> (x * (x + 1)) ÷ 2, blockSizes) != P.n
        if verbose
            @error("Dimensions do not match up. Rounding error (try different epsilons and/or try again) or not block-diagonalizable over the reals (try parameter complex = true).")
            @show sum(x -> (x * (x + 1)) ÷ 2, blockSizes)
            @show P.n
        end
        return nothing
    elseif complex && sum(x -> x ^ 2, blockSizes) != P.n
        if verbose
            @error("Dimensions do not match up. Probably a rounding error (try different epsilons and/or try again).")
            @show sum(x -> x ^ 2, blockSizes)
            @show P.n
        end
        return nothing
    end

    verbose && println("Determining the algebra-isomorphism...")

    @time begin
    reducedQis = Matrix{T}[]
    for i in eachindex(uniqueKs)
        Ki = uniqueKs[i]
        bs = blockSizes[i]

        QKi = hcat((view(Q, :, el) for (i, el) in enumerate(indEV) if K[i] == Ki)...)
        getRandomMatrix!(r, A, P; complex)
        C = QKi' * A * QKi # Symmetric needed in general?
        QKi3 = zeros(T, size(C))

        mult = countEV[Ki]

        QKi3[1:mult, 1:mult] .= I(mult)
        for j in 2:bs
            ind = mult*(j-1) .+ (1:mult)
            QKi3[ind, ind] .= inv(C[1:mult, ind])
            QKi3[ind, ind] ./= norm(QKi3[mult*(j-1)+1, ind])
        end

        Per = zeros(T, size(C))
        for i in 1:bs, j in 1:mult
            Per[i+bs*(j-1), j+mult*(i-1)] = 1
        end

        reducedQi = (QKi * QKi3 * Per')[:, 1:bs]

        push!(reducedQis, reducedQi)
    end
    end

    verbose && println("Calculating image of the basis of the algebra...")

    @time begin
    # blockDiagonalization = [[roundToZero!(B) for B in [Qi' * P * Qi for Qi in reducedQis]] for P in [P.P .== i for i in 1:P.n]]
    blockDiagonalization = [[zeros(T, bs, bs) for bs in blockSizes] for _ in 1:P.n]
    tmp = [Matrix{T}(undef, bs, bs) for bs in blockSizes]
    for x in axes(P.P, 1), y in axes(P.P, 2)
        PP = P.P[x, y]
        if PP != 0
            for (i, Qi) in enumerate(reducedQis)
                mul!(tmp[i], Qi[x, :], Qi[y, :]') # might need to swap x and y for non symmetric matrices
                blockDiagonalization[PP][i] .+= tmp[i]
            end
        end
    end
    broadcast.(roundToZero!, blockDiagonalization)

    end
    return (blkSizes = blockSizes, blks = blockDiagonalization)
end
# move the type unstability to this function, also avoid breaking the old syntax
function blockDiagonalize(P::Partition, verbose = true; epsilon = Base.rtoldefault(Float64), complex = false)
    if !complex
        return blockDiagonalize(Float64, P, verbose; epsilon)
    else
        return blockDiagonalize(ComplexF64, P, verbose; epsilon)
    end
end

end
