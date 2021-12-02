# A slightly more advanced example how to reduce and solve a given SDP in standard form. Example application in QuadraticAssignmentProblems.jl

using LinearAlgebra
using SDPSymmetryReduction
using JuMP
using Hypatia
using SparseArrays

function reduceAndSolve(C, A, b, objSense = MathOptInterface.MAX_SENSE, verbose = false, complex = false, limitSize = 3000)
    tmd = @timed admPartSubspace(C, A, b, verbose)
    P = tmd.value
    jordanTime = tmd.time

    if P.n <= limitSize

        tmd = @timed blockDiagonalize(P, verbose; complex = complex)
        blkD = tmd.value
        blkDTime = tmd.time

        if blkD === nothing
            # Either random/rounding error, or complex numbers needed
            return nothing
        end

        # solve with solver of choice
        m = nothing
        if verbose
            m = Model(Hypatia.Optimizer)
        else
            m = Model(optimizer_with_attributes(Hypatia.Optimizer, "MSK_IPAR_LOG" => 0))
        end

        # >= 0 because the SDP-matrices should be entry-wise nonnegative
        x = @variable(m, x[1:P.n] >= 0)

        PMat = hcat([sparse(vec(P.P .== i)) for i = 1:P.n]...)

        # Reduce the number of constraints
        newConstraints = Float64.(hcat(A * PMat, b))
        newConstraints = sparse(svd(Matrix(newConstraints)').U[:, 1:rank(newConstraints)]')
        droptol!(newConstraints, 1e-8)

        newA = newConstraints[:, 1:end-1]
        newB = newConstraints[:, end]
        newC = C' * PMat

        @constraint(m, newA * x .== newB)
        @objective(m, objSense, newC * x)

        for i = 1:length(blkD[1])
            blkExpr =
                x[1] .* (
                    complex ?
                    [
                        real(blkD[2][1][i]) -imag(blkD[2][1][i])
                        imag(blkD[2][1][i]) real(blkD[2][1][i])
                    ] :
                    blkD[2][1][i]
                )
            for j = 2:P.n
                add_to_expression!.(
                    blkExpr,
                    x[j] .* (
                        complex ?
                        [
                            real(blkD[2][j][i]) -imag(blkD[2][j][i])
                            imag(blkD[2][j][i]) real(blkD[2][j][i])
                        ] :
                        blkD[2][j][i]
                    ),
                )
            end
            if size(blkExpr, 1) > 1
                @constraint(m, blkExpr in PSDCone())
            else
                @constraint(m, blkExpr .>= 0)
            end
        end

        tmd = @timed optimize!(m)
        optTime = tmd.time

        if Int64(termination_status(m)) != 1
            @show termination_status(m)
            @error("Solve error.")
        end
        return (
            jTime = jordanTime,
            blkTime = blkDTime,
            solveTime = optTime,
            optVal = newC * value.(x),
            blkSize = blkD[1],
            originalSize = size(P.P, 1),
            newSize = P.n
        )
    end
    return (
        jTime = jordanTime,
        blkTime = 0,
        solveTime = 0,
        optVal = 0,
        blkSize = 0,
        originalSize = size(P.P, 1),
        newSize = P.n
    )

end
