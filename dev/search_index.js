var documenterSearchIndex = {"docs":
[{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"EditURL = \"https://github.com/DanielBrosch/SDPSymmetryReduction.jl/blob/master/docs/src/examples/ErdosRenyiThetaFunction.jl\"","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/#The-Theta'-function-of-Erdos-Renyi-graphs","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"","category":"section"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"Let q be an odd prime, and let V = mathrmGF(q)^3 be a three dimensional vector space over the finite field of order q. The set of one dimensional subspaces, i.e. the projective plane, of V is denoted by mathrmPG(2q). There are q^2+q+1 such subspaces, which are the vertices of the Erdos-Renyi graph mathrmER(q). Two vertices are adjacent if they are distinct and orthogonal, i.e. for two representing vectors x and y we have x^Ty=0. We are interested in the size of a maximum stable set of these graphs, specifically upper bounds for this value. Note that these are not the equally named random graphs.","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"using LinearAlgebra #hide\nq = 7\nPG2q = vcat([[0, 0, 1]],\n            [[0, 1, b] for b = 0:q-1],\n            [[1, a, b] for a = 0:q-1 for b = 0:q-1])\nAdj = [x' * y % q == 0 && x != y for x in PG2q, y in PG2q]\nsize(Adj)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"using Plots #hide\nspy(Adj)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/#The-Theta'-function","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function","text":"","category":"section"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"The Theta'-function vartheta(G) of a graph G=(VE) is such an upper bound, based on semidefinite programming:","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"vartheta(G)coloneqq suplangle XJrangle  langle XArangle = 0 Xsucccurlyeq 0 Xgeq 0","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"In vectorized standard form this is simply","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"N = length(PG2q)\nC = ones(N^2)\nA = vcat(vec(Adj)', vec(Matrix{Float64}(I, N, N))')\nb = [0.0, 1.0];\nnothing #hide","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/#Determining-the-symmetry-reduction","page":"The Theta'-function of Erdos-Renyi graphs","title":"Determining the symmetry reduction","text":"","category":"section"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"We can now apply the Jordan reduction method to the problem. First, we need to determine an (optimal) admissible subspace.","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"using SDPSymmetryReduction\nP = admPartSubspace(C, A, b, true)\nP.n","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"Running admPartSubspace returns a Partition object. P.n are the number of orbits (and thus variables), and P.P is a matrix with integer values from 1 trough P.n. Here, P.P looks like this (different color shades = different orbits):","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"heatmap(reverse(P.P, dims=1)) #hide","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"Now we can block-diagonalize the algebra (numerically)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"blkD = blockDiagonalize(P, true);\nnothing #hide","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/#Building-the-reduced-SDP","page":"The Theta'-function of Erdos-Renyi graphs","title":"Building the reduced SDP","text":"","category":"section"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"Since blkD.blks[i] is the block-diagonalized image of P.P .== i, we obtain the new, symmetry reduced SDP by","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"using SparseArrays\nPMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)\nnewA = A * PMat\nnewB = b\nnewC = C' * PMat;\nnothing #hide","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/#Solving-the-SDP-with-JuMP-and-CSDP","page":"The Theta'-function of Erdos-Renyi graphs","title":"Solving the SDP with JuMP and CSDP","text":"","category":"section"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"using JuMP, CSDP\nm = Model(CSDP.Optimizer)\n\n# Initialize variables corresponding parts of the partition P\n# >= 0 because the original SDP-matrices are entry-wise nonnegative\nx = @variable(m, x[1:P.n] >= 0)\n\n@constraint(m, newA * x .== newB)\n@objective(m, Max, newC * x)\n\npsdBlocks = sum(blkD.blks[i] .* x[i] for i = 1:P.n)\nfor blk in psdBlocks\n    if size(blk, 1) > 1\n        @constraint(m, blk in PSDCone())\n    else\n        @constraint(m, blk .>= 0)\n    end\nend\n\noptimize!(m)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"termination_status(m)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"objective_value(m)","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"","category":"page"},{"location":"examples/ErdosRenyiThetaFunction/","page":"The Theta'-function of Erdos-Renyi graphs","title":"The Theta'-function of Erdos-Renyi graphs","text":"This page was generated using Literate.jl.","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"EditURL = \"https://github.com/DanielBrosch/SDPSymmetryReduction.jl/blob/master/docs/src/examples/QuadraticAssignmentProblems.jl\"","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Symmetry-reducting-a-strong-relaxation-of-the-quadratic-assigment-problem","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"Here, we are going to show how to load a QAP from QABLib, formulate a strong semidefinite relaxation of it, symmetry reduce it, and finally solve it.","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Quadratic-assigment-problems","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Quadratic assigment problems","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"QAPs are given by two quadratic matrices A and B. The objective is to permute the rows and columns of B, such that the inner product between the matrices is minimized.","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"mathrmQAP(AB) = min_phiin S_n sum_ij=1^n a_ijb_phi(i)phi(j)","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"QAPs are notoriously hard to solve exactly, but there exist strong polynomial time relaxations, such as the following semidefinite programming relaxation:","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"beginaligned\nminenspace  langle Botimes A Yrangle\nmathrmstenspace  langle I_notimes E_jjYrangle=1 text for jin n\n langle E_jjotimes I_nYrangle=1 text for jin n\n langle I_notimes (J_n-I_n)+(J_n-I_n)otimes I_nYrangle =0 \n langle J_n^2Yrangle = n^2\n Yin D^n^2\nendaligned","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"But in practice this relaxation is often too big to be solved directly.","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Loading-the-data-of-a-QAP","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Loading the data of a QAP","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"using SparseArrays, LinearAlgebra\n\nfile = joinpath(@__DIR__, \"esc16j.dat\")\ndata = open(file) do file\n    read(file, String)\nend\ndata = split(data, [' ', '\\n', '\\r'], keepempty = false)\n\nn = parse(Int64, data[1])\nA = zeros(Int64, n, n)\nB = zeros(Int64, n, n)\n\npos = 2\nfor x = 1:n\n    for y = 1:n\n        A[x, y] = parse(Int64, data[pos])\n        global pos += 1\n    end\nend\nfor x = 1:n\n    for y = 1:n\n        B[x, y] = parse(Int64, data[pos])\n        global pos += 1\n    end\nend","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Building-the-SDP-(in-vectorized-standard-form)","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Building the SDP (in vectorized standard form)","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"n = size(A, 1)\n\n# Objective\nCPrg = sparse(kron(B, A))\n\nIn = sparse(I, n, n)\nJn = ones(n, n)\n\n# Vectorised constraint matrices as rows of large matrix APrg\nAPrg = spzeros(2n + 1, n^4)\nbPrg = zeros(2n + 1)\ncurrentRow = 1\n\nfor j = 1:n\n    Ejj = spzeros(n, n)\n    Ejj[j, j] = 1.0\n    APrg[currentRow, :] = vec(kron(In, Ejj))\n    bPrg[currentRow] = 1\n    global currentRow += 1\n    # Last constraint is linearly dependent on others\n    if (j < n)\n        APrg[currentRow, :] = vec(kron(Ejj, In))\n        bPrg[currentRow] = 1\n        global currentRow += 1\n    end\nend\n\nAPrg[currentRow, :] = vec(kron(In, Jn - In) + kron(Jn - In, In))\nbPrg[currentRow] = 0\ncurrentRow += 1\nAPrg[currentRow, :] = vec(ones(n^2, n^2))\nbPrg[currentRow] = n^2\n\nCPrg = sparse(vec(0.5 * (CPrg + CPrg')));\nnothing #hide","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Symmetry-reducing-the-SDP","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducing the SDP","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"We first determine an optimal admissible partition subspace","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"using SDPSymmetryReduction\nP = admPartSubspace(CPrg, APrg, bPrg, true)\nP.n","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"And then we block-diagonalize it","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"blkD = blockDiagonalize(P, true);\nnothing #hide","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Determining-the-coefficients-of-the-reduced-SDP","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Determining the coefficients of the reduced SDP","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)\nnewA = APrg * PMat\nnewB = bPrg\nnewC = CPrg' * PMat;\nnothing #hide","category":"page"},{"location":"examples/QuadraticAssignmentProblems/#Solving-the-reduced-SDP-with-JuMP-and-CSDP","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Solving the reduced SDP with JuMP and CSDP","text":"","category":"section"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"using JuMP, CSDP\nm = Model(CSDP.Optimizer)\n\n# Initialize variables corresponding parts of the partition P\n# >= 0 because the original SDP-matrices are entry-wise nonnegative\nx = @variable(m, x[1:P.n] >= 0)\n\n@constraint(m, newA * x .== newB)\n@objective(m, Max, newC * x)\n\npsdBlocks = sum(blkD.blks[i] .* x[i] for i = 1:P.n)\nfor blk in psdBlocks\n    if size(blk, 1) > 1\n        @constraint(m, blk in PSDCone())\n    else\n        @constraint(m, blk .>= 0)\n    end\nend\n\noptimize!(m)","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"termination_status(m)","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"objective_value(m)","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"","category":"page"},{"location":"examples/QuadraticAssignmentProblems/","page":"Symmetry reducting a strong relaxation of the quadratic assigment problem","title":"Symmetry reducting a strong relaxation of the quadratic assigment problem","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SDPSymmetryReduction","category":"page"},{"location":"#SDPSymmetryReduction","page":"Home","title":"SDPSymmetryReduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SDPSymmetryReduction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SDPSymmetryReduction]","category":"page"},{"location":"#SDPSymmetryReduction.admPartSubspace-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractMatrix{T}, AbstractVector{T}}, Tuple{AbstractVector{T}, AbstractMatrix{T}, AbstractVector{T}, Bool}} where T<:AbstractFloat","page":"Home","title":"SDPSymmetryReduction.admPartSubspace","text":"admPartSubspace(C::Vector{T}, A::Matrix{T}, b::Vector{T}, verbose::Bool = false)\n\nCompute the smallest admissible partion subspace for the SDP infsupdot(Cx) Ax = b mathrmMat(x) textPSDDNN\n\n\n\n\n\n","category":"method"},{"location":"#SDPSymmetryReduction.blockDiagonalize","page":"Home","title":"SDPSymmetryReduction.blockDiagonalize","text":"blockDiagonalize(P::Partition, verbose = true; epsilon = 1e-8, complex = false)\n\nCompute the block-diagonalization of the (Jordan)-algebra given by the Partition P. May fail if complex = false, since a real block diagonalization does not always exist.\n\n\n\n\n\n","category":"function"},{"location":"examples/ReduceAndSolveJuMP/","page":"General example","title":"General example","text":"EditURL = \"https://github.com/DanielBrosch/SDPSymmetryReduction.jl/blob/master/docs/src/examples/ReduceAndSolveJuMP.jl\"","category":"page"},{"location":"examples/ReduceAndSolveJuMP/#General-example","page":"General example","title":"General example","text":"","category":"section"},{"location":"examples/ReduceAndSolveJuMP/","page":"General example","title":"General example","text":"This function takes an SDP in standard form, reduces it, formulates it as (hermitian) SDP, and solves it with JuMP","category":"page"},{"location":"examples/ReduceAndSolveJuMP/","page":"General example","title":"General example","text":"using LinearAlgebra\nusing SDPSymmetryReduction\nusing JuMP\nusing CSDP\nusing SparseArrays\n\nfunction reduceAndSolve(C, A, b;\n    objSense = MathOptInterface.MAX_SENSE,\n    verbose = false,\n    complex = false,\n    limitSize = 3000)\n\n    tmd = @timed admPartSubspace(C, A, b, verbose)\n    P = tmd.value\n    jordanTime = tmd.time\n\n    if P.n <= limitSize\n\n        tmd = @timed blockDiagonalize(P, verbose; complex = complex)\n        blkD = tmd.value\n        blkDTime = tmd.time\n\n        if blkD === nothing\n            # Either random/rounding error, or complex numbers needed\n            return nothing\n        end\n\n        # solve with solver of choice\n        m = nothing\n        if verbose\n            m = Model(CSDP.Optimizer)\n        else\n            m = Model(optimizer_with_attributes(CSDP.Optimizer, \"MSK_IPAR_LOG\" => 0))\n        end\n\n        # >= 0 because the SDP-matrices should be entry-wise nonnegative\n        x = @variable(m, x[1:P.n] >= 0)\n\n        PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)\n\n        # Reduce the number of constraints\n        newConstraints = Float64.(hcat(A * PMat, b))\n        newConstraints = sparse(svd(Matrix(newConstraints)').U[:, 1:rank(newConstraints)]')\n        droptol!(newConstraints, 1e-8)\n\n        newA = newConstraints[:, 1:end-1]\n        newB = newConstraints[:, end]\n        newC = C' * PMat\n\n        @constraint(m, newA * x .== newB)\n        @objective(m, objSense, newC * x)\n\n        for i = 1:length(blkD[1])\n            blkExpr =\n                x[1] .* (\n                    complex ?\n                    [\n                        real(blkD[2][1][i]) -imag(blkD[2][1][i])\n                        imag(blkD[2][1][i]) real(blkD[2][1][i])\n                    ] :\n                    blkD[2][1][i]\n                )\n            for j = 2:P.n\n                add_to_expression!.(\n                    blkExpr,\n                    x[j] .* (\n                        complex ?\n                        [\n                            real(blkD[2][j][i]) -imag(blkD[2][j][i])\n                            imag(blkD[2][j][i]) real(blkD[2][j][i])\n                        ] :\n                        blkD[2][j][i]\n                    ),\n                )\n            end\n            if size(blkExpr, 1) > 1\n                @constraint(m, blkExpr in PSDCone())\n            else\n                @constraint(m, blkExpr .>= 0)\n            end\n        end\n\n        tmd = @timed optimize!(m)\n        optTime = tmd.time\n\n        if Int64(termination_status(m)) != 1\n            @show termination_status(m)\n            @error(\"Solve error.\")\n        end\n        return (\n            jTime = jordanTime,\n            blkTime = blkDTime,\n            solveTime = optTime,\n            optVal = newC * value.(x),\n            blkSize = blkD[1],\n            originalSize = size(P.P, 1),\n            newSize = P.n\n        )\n    end\n    return (\n        jTime = jordanTime,\n        blkTime = 0,\n        solveTime = 0,\n        optVal = 0,\n        blkSize = 0,\n        originalSize = size(P.P, 1),\n        newSize = P.n\n    )\n\nend","category":"page"},{"location":"examples/ReduceAndSolveJuMP/","page":"General example","title":"General example","text":"","category":"page"},{"location":"examples/ReduceAndSolveJuMP/","page":"General example","title":"General example","text":"This page was generated using Literate.jl.","category":"page"}]
}