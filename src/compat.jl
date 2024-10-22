roundToZero(f::Number) = clamptol(f)
roundToZero!(a::AbstractArray) = clamptol_rec!(a)

function orthProject(A::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    return project_colspace(v, A)
end

part(M) = Partition(M)
coarsestPart(P, Q) = refine(P, Q)
rndPart(P) = randomize(Float64, P)

roundMat(M) = M .= clamptol.(round.(M, sigdigits=5))

function projectAndRound(M, A; round=true)
    v = vec(M)
    v .-= orthProject(A, v)
    if round
        roundMat(v)
    end
    return Float64.(reshape(v, size(M)...))
end

admPartSubspace(C, A, b, verbose=false) = admissible_subspace(C, A, b, verbose)

# move the type unstability to this function, also avoid breaking the old syntax
function blockDiagonalize(P, verbose=true; epsilon=Base.rtoldefault(Float64), complex=false)
    if !complex
        return blockDiagonalize(Float64, P, verbose; epsilon)
    else
        P = unSymmetrize(P)
        return blockDiagonalize(ComplexF64, P, verbose; epsilon)
    end
end

"""
    blockDiagonalize(P::Partition, verbose = true; epsilon = 1e-8, complex = false)

Determines a block-diagonalization of a (Jordan)-algebra given by a partition `P` using a randomized algorithm. `blockDiagonalize(P)` returns a real block-diagonalization `blkd`, if it exists, otherwise `nothing`.

`blockDiagonalize(P; complex = true)` returns the same, but with complex valued matrices, and should be used if no real block-diagonalization was found. To use the complex matrices practically, remember that a Hermitian matrix `A` is positive semidefinite iff `[real(A) -imag(A); imag(A) real(A)]` is positive semidefinite.

## Output

* `blkd.blkSizes` is an integer array of the sizes of the blocks.
* `blkd.blks` is an array of length `P.n` containing arrays of (real/complex) matrices of sizes `blkd.blkSizes`. I.e. `blkd.blks[i]` is the image of the basis element `P.P .== i`.
"""
function blockDiagonalize(
    ::Type{T},
    P,
    verbose=true;
    epsilon=Base.rtoldefault(real(T))
) where {T}

    Q_hat = try
        diagonalize(T, P, verbose; epsilon=epsilon)
    catch err
        if err isa InvalidDecompositionField
            @error err
        else
            rethrow(err)
        end
        return nothing
    end

    # throws DimensionMismatch if appropriate
    check_block_sizes(T, Q_hat, P, verbose)

    verbose && @info "Calculating image of the basis of the algebra..."
    t = @timed basis_img = basis_image(Q_hat, P)

    verbose && @info sprint(timed_print, t)

    return (blkSizes=[size(q, 2) for q in Q_hat], blks=basis_img)
end
