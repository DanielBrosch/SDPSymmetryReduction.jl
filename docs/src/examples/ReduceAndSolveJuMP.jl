# # General example
# This function takes an SDP in standard form, reduces it, formulates it as (hermitian) SDP, and solves it with JuMP

using LinearAlgebra
using SDPSymmetryReduction
using JuMP
using CSDP
using SparseArrays

function reduceAndSolve(C, A, b; 
    objSense = MathOptInterface.MAX_SENSE, 
    verbose = false, 
    complex = false, 
    limitSize = 3000)
    
    tmd = @timed admissible_subspace(C, A, b; verbose=verbose)
    P = tmd.value
    jordanTime = tmd.time

    if dim(P) <= limitSize

        tmd = @timed blockDiagonalize(P, verbose; complex = complex)
        blkD = tmd.value
        blkDTime = tmd.time

        if blkD === nothing
            ## Either random/rounding error, or complex numbers needed
            return nothing
        end

        ## solve with solver of choice
        m = nothing
        if verbose
            m = Model(CSDP.Optimizer)
        else
            m = Model(optimizer_with_attributes(CSDP.Optimizer, "MSK_IPAR_LOG" => 0))
        end

        ## >= 0 because the SDP-matrices should be entry-wise nonnegative
        x = @variable(m, x[1:dim(P)] >= 0)

        PMat = hcat([sparse(vec(P.matrix .== i)) for i = 1:dim(P)]...)

        ## Reduce the number of constraints
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
            for j = 2:dim(P)
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
            originalSize = size(P.matrix, 1),
            newSize = dim(P)
        )
    end
    return (
        jTime = jordanTime,
        blkTime = 0,
        solveTime = 0,
        optVal = 0,
        blkSize = 0,
        originalSize = size(P.matrix, 1),
        newSize = dim(P)
    )

end
