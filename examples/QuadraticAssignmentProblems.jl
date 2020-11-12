## Load a QAP and determine an admissible subspace of its SDP relaxation
using SDPSymmetryReduction

using SparseArrays
using LinearAlgebra
using DelimitedFiles

using JuMP
using MosekTools


# Symmetry reduces the given QAP, returns optimal admissible partition subspace
function reduceQAP(A, B, C = zeros(size(A)))
    prg = generateSDP(A, B, C)
    P = admPartSubspace(prg...)
    return P
end

# Load a QAP from a file (QAPLib format)
function loadQAP(file)
    data = open(file) do file
        read(file, String)
    end
    data = split(data, [' ', '\n', '\r'], keepempty = false)

    n = parse(Int64, data[1])
    A = zeros(Int64, n, n)
    B = zeros(Int64, n, n)

    pos = 2
    for x = 1:n
        for y = 1:n
            A[x, y] = parse(Int64, data[pos])
            pos = pos + 1
        end
    end
    for x = 1:n
        for y = 1:n
            B[x, y] = parse(Int64, data[pos])
            pos = pos + 1
        end
    end

    return (A = A, B = B)
end

#Formulating the QAP DNN relaxation with vectorised matrices in conic form
function generateSDP(A, B, C = zeros(size(A)))
    n = size(A, 1)

    #Objective
    CPrg = sparse(kron(B, A) + Diagonal(vec(C)))

    In = sparse(I, n, n)
    Jn = ones(n, n)

    tmp = kron(In, Jn - In) + kron(Jn - In, In)




    #Vectorised condition matrices as rows of large matrix APrg
    APrg = spzeros(2n + 1, n^4)
    b = zeros(2n + 1)
    currentRow = 1


    for j = 1:n
        Ejj = spzeros(n, n)
        Ejj[j, j] = 1.0
        APrg[currentRow, :] = vec(kron(In, Ejj))
        b[currentRow] = 1
        currentRow += 1
        # Last condition is linearly dependent on others
        if (j < n)
            APrg[currentRow, :] = vec(kron(Ejj, In))
            b[currentRow] = 1
            currentRow += 1
        end
    end

    APrg[currentRow, :] = vec(kron(In, Jn - In) + kron(Jn - In, In))
    b[currentRow] = 0
    currentRow += 1
    APrg[currentRow, :] = vec(ones(n^2, n^2))
    b[currentRow] = n^2

    CPrg = sparse(vec(0.5 * (CPrg + CPrg')))

    return (C = CPrg, A = APrg, b = b)
end


## Solving an a relaxation of an example QAP from QAPLib

include("ReduceAndSolveJuMP.jl")
qap = loadQAP("examples\\esc16j.dat")
prg = generateSDP(qap.A, qap.B)
@show reduceAndSolve(prg.C, prg.A, prg.b, MathOptInterface.MIN_SENSE, true)
