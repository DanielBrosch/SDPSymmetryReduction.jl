module SDPSymmetryReduction

using SparseArrays
using LinearAlgebra

export Partition, admPartSubspace, blockDiagonalize

# Stores a partition of [m]×[m] in a single matrix
# Entries of P should always be 1,…,n
struct Partition
    n::Int64 # Number of parts
    P::Matrix{Int64} # Matrix with entries 1,...,n
end

# Round numbers near zero to zero
function roundToZero(f::Number)
    if abs(f) < 1e-8
        return 0
    end
    return f
end

# Create a partition from the unique entries of M
function part(M)
    u = unique(M)
    filter!(e -> e ≠ 0, u)
    d = Dict([(u[i], i) for i in eachindex(u)])
    d[0] = 0

    return Partition(size(u, 1), [d[i] for i in M])
end

# Find the coarsest partition refining P1 and P2
function coarsestPart(P1::Partition, P2::Partition)
    return part(P1.P * (P2.n + 1) + P2.P)
end

# Returns a random linear combination in the partition space
function rndPart(P::Partition, orthog = false)
    r = [rand() for i = 1:P.n+1]
    r[1] = 0
    if orthog
        for i = 1:P.n
            r[i+1] /= count(x -> x == i, P.P)
        end
    end
    return [r[i+1] for i in P.P]
end

# Rounds the matrix, sets entries near 0.0 to 0.0
function roundMat(M)
    tmp = round.(M, sigdigits = 5)
    tmp = roundToZero.(tmp)
    return tmp
end

# Project v orthogonally to the span of columns of A
function orthProject(A, v)
    return A * ((A' * A) \ (A' * v))
end

# Projects M and rounds the matrix afterwards
function projectAndRound(M, A, round = true)
    n = Int64(sqrt(length(M)))
    tmp = vec(Matrix(M))
    tmp = tmp - orthProject(A, tmp)
    if round
        tmp = roundMat(tmp)
    end
    tmp = reshape(tmp, n, n)
    return Float64.(tmp)
end

"""
    admPartSubspace(C, A, b, verbose = false)

Compute smallest admissible partion subspace for the SDP inf/sup{C'*x, A*x = b, Mat(x) PSD/DNN}.
"""
function admPartSubspace(C, A, b, verbose = false)

    n = Int(sqrt(Float64(size(C, 1))))

    A = Float64.(A)

    verbose && print("\nStarting the reduction. Original dimension: $(Int64((n^2+n)/2))\n",)

    C = roundMat(C)
    CL = C - orthProject(A', Vector(C))
    CL = reshape(roundMat(CL), n, n)
    CL = Matrix(Symmetric(0.5 * (CL + CL')))

    X0 = sparse(qr(sparse(A)) \ Vector(b))

    X0Sym = vec(0.5 * (reshape(X0, n, n) + reshape(X0, n, n)'))
    X0Lp = orthProject(A', X0Sym)
    X0Lp = roundMat(X0Lp)
    X0Lp = Matrix(Symmetric(reshape(X0Lp, n, n)))

    # Initialize P as the coarsest partition refining the partitions of two matrices
    P = coarsestPart(part(X0Lp), part(CL))
    dim = P.n

    it = 0

    # Iterate until converged
    while true
        it += 1

        verbose && print("Iteration $it, Current dimension: $dim\n",)

        # Add a random projection to L
        randomizedP = rndPart(P)
        BL = projectAndRound(randomizedP, A')
        P = coarsestPart(P, part(BL))

        # If the last projection didn't change the partition, reuse the random linear combination
        if (dim != P.n)
            randomizedP = rndPart(P)
        end

        # Add random square
        P2 = randomizedP^2
        P2 = roundMat(P2)

        P = coarsestPart(P, part(P2))

        # Check if converged (not changed or reached full partition)
        if (dim == P.n) | (P.n == Int64((n^2 + n) / 2))
            dim = P.n
            break
        end

        dim = P.n
    end

    verbose &&
        print("$it Total iterations, Final dimension: $dim, Old dimension: $(Int64((n^2+n)/2))\n")
    return P
end

# WL algorithm to "un-symmetrize" the Jordan algebra
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


"""
    blockDiagonalize(P::Partition, verbose = true; epsilon = 1e-8, complex = false)

Compute the block-diagonalization of the (Jordan)-algebra given by the Partition P. May fail if complex = false, since a real block diagonalization does not always exist.
"""
function blockDiagonalize(P::Partition, verbose = true; epsilon = 1e-8, complex = false)
    P2 = P
    if complex
        P2 = unSymmetrize(P)
    end

    function getRandomMatrix()
        if !complex
            return rndPart(P)
        else
            tmp = rndPart(P2) + im * rndPart(P2)
            return tmp + tmp'
        end
    end

    verbose && println("Determining block sizes...")

    A = getRandomMatrix()
    F = eigen(A)
    Q = F.vectors

    # split by eigenvalues
    roundedEV = round.(F.values, digits = 10)
    uniqueEV = unique(roundedEV)
    countEV = [count(roundedEV .== u) for u in uniqueEV]

    QSplit = [Q[:, [i for i = 1:length(roundedEV) if roundedEV[i] == u]] for u in uniqueEV]

    PSplit = [P.P .== i for i = 1:P.n]

    K = collect(1:length(uniqueEV))
    tmp = getRandomMatrix()
    for i = 1:length(uniqueEV)
        for j = i:length(uniqueEV)
            if K[i] != K[j] && countEV[i] == countEV[j]
                if any(x -> x >= epsilon, abs.(QSplit[i]' * tmp * QSplit[j]))
                    K[K.==K[j]] .= K[i]
                end
            end
        end
    end

    verbose && println("Block sizes are $(sort!([count(K.==Ki) for Ki in unique(K)]))")

    if !complex &&
       sum([count(K .== Ki) * (count(K .== Ki) + 1) / 2 for Ki in unique(K)]) != P.n
        if verbose
            @error("Dimensions do not match up. Rounding error (try different epsilons and/or try again) or not block-diagonalizable over the reals (try parameter complex = true).")
            @show sum([count(K .== Ki) * (count(K .== Ki) + 1) / 2 for Ki in unique(K)])
            @show P.n
        end
        return nothing
    elseif complex && sum([count(K .== Ki)^2 for Ki in unique(K)]) != P2.n
        if verbose
            @error("Dimensions do not match up. Probably a rounding error (try different epsilons and/or try again).")
            @show sum([count(K .== Ki)^2 for Ki in unique(K)])
            @show P2.n
        end
        return nothing
    end


    verbose && println("Determining the algebra-isomorphism...")

    uniqueKs = unique(K)
    reducedQis = []
    for Ki in uniqueKs
        countKi = count(K .== Ki)

        QKi = hcat([QSplit[i] for i = 1:length(uniqueEV) if K[i] == Ki]...)
        B1 = Symmetric(QKi' * getRandomMatrix() * QKi)
        QKi3 = zeros(complex ? Complex{Float64} : Float64, size(B1))

        mult = countEV[Ki]

        for j = 1:countKi
            if j == 1

                QKi3[
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                ] = Matrix(I, mult, mult)
            else
                QKi3[
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                ] =
                    B1[
                        1:mult,
                        (mult*(j-1)+1):(mult*(j-1)+mult),
                    ]^(-1)

                QKi3[
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                ] ./= norm(QKi3[
                    (mult*(j-1)+1),
                    (mult*(j-1)+1):(mult*(j-1)+mult),
                ])

            end
        end

        Per = zeros(size(B1))
        for i = 1:countKi
            for j = 1:mult
                Per[i+countKi*(j-1), j+mult*(i-1)] = 1
            end
        end

        reducedQi = (QKi*QKi3*Per')[:, 1:countKi]

        push!(reducedQis, reducedQi)

    end

    verbose && println("Calculating image of the basis of the algebra...")
    blockDiagonalization = [
        [
            (
                complex ?
                real(B) .* (abs.(real(B)) .>= epsilon) +
                im * imag(B) .* (abs.(imag(B)) .>= epsilon) :
                B .* (abs.(B) .>= epsilon)
            ) for B in [Qi' * P * Qi for Qi in reducedQis]
        ] for P in PSplit
    ]

    blockSizes = [size(b)[1] for b in blockDiagonalization[1]]
    return (blkSizes = blockSizes, blks = blockDiagonalization)
end


end
