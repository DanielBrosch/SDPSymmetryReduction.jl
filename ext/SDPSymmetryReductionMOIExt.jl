module SDPSymmetryReductionMOIExt

import LinearAlgebra
import MathOptInterface as MOI
import SDPSymmetryReduction as SR
import SparseArrays

const PSD = MOI.PositiveSemidefiniteConeTriangle
const SAF = MOI.ScalarAffineFunction{Float64}
const VAF = MOI.VectorAffineFunction{Float64}

MOI.Utilities.@product_of_sets(PSDCones, PSD)

# Cache storing the problem in geometric conic form
const Cache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{Float64},
        PSDCones{Float64},
    },
}

# The original problem is
#   min/max  c'x + constant
#   s.t.     A x + a ∈ PositiveSemidefiniteConeTriangle(side_dimension)
# where `A x + a` is the vectorization of the upper triangle of the matrix
# `F₀ + ∑ᵢ xᵢ Fᵢ`.
mutable struct Optimizer <: MOI.AbstractOptimizer
    optimizer_constructor::Any
    verbose::Bool
    tol::Float64
    silent::Bool
    side_dimension::Int
    sense::MOI.OptimizationSense
    c::Vector{Float64}
    constant::Float64
    A::SparseArrays.SparseMatrixCSC{Float64,Int}
    a::Vector{Float64}
    inner::Any
    # Results, `primal` is `x` and `dual` is the dual of the semidefinite
    # constraint
    primal::Union{Nothing,Vector{Float64}}
    dual::Union{Nothing,Vector{Float64}}
    solve_time::Float64
    function Optimizer(optimizer_constructor; verbose::Bool, tol::Float64)
        return new(
            optimizer_constructor,
            verbose,
            tol,
            false,
            0,
            MOI.FEASIBILITY_SENSE,
            Float64[],
            0.0,
            SparseArrays.spzeros(0, 0),
            Float64[],
            nothing,
            nothing,
            nothing,
            NaN,
        )
    end
end

function SR.Optimizer(optimizer_constructor; verbose::Bool = false, tol = 1e-6)
    return Optimizer(optimizer_constructor; verbose, tol = Float64(tol))
end

function MOI.empty!(model::Optimizer)
    model.side_dimension = 0
    model.sense = MOI.FEASIBILITY_SENSE
    empty!(model.c)
    model.constant = 0.0
    model.A = SparseArrays.spzeros(0, 0)
    empty!(model.a)
    model.inner = nothing
    model.primal = nothing
    model.dual = nothing
    model.solve_time = NaN
    return
end

function MOI.is_empty(model::Optimizer)
    return model.side_dimension == 0 && isempty(model.c) &&
           model.sense == MOI.FEASIBILITY_SENSE && isnothing(model.inner)
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    if isnothing(model.inner)
        return "SDPSymmetryReduction"
    end
    return "SDPSymmetryReduction with $(MOI.get(model.inner, MOI.SolverName()))"
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,SAF}},
)
    return true
end

MOI.supports_constraint(::Optimizer, ::Type{VAF}, ::Type{PSD}) = true

function _check_supported(model::Optimizer, src::MOI.ModelLike)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(model, F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if !(attr isa MOI.ConstraintName)
                throw(MOI.UnsupportedAttribute(attr))
            end
        end
    end
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if !(attr isa MOI.Name || MOI.supports(model, attr))
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if !(attr isa MOI.VariableName)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    _check_supported(dest, src)
    num_psd = MOI.get(src, MOI.NumberOfConstraints{VAF,PSD}())
    if num_psd != 1
        error(
            "SDPSymmetryReduction.Optimizer only supports problems with " *
            "exactly one `MOI.VectorAffineFunction`-in-" *
            "`MOI.PositiveSemidefiniteConeTriangle` constraint, the problem " *
            "has $num_psd.",
        )
    end
    cache = Cache()
    index_map = MOI.copy_to(cache, src)
    dest.A = convert(
        SparseArrays.SparseMatrixCSC{Float64,Int},
        cache.constraints.coefficients,
    )
    dest.a = copy(cache.constraints.constants)
    dest.side_dimension =
        MOI.Utilities.side_dimension_for_vectorized_dimension(size(dest.A, 1))
    dest.sense = MOI.get(cache, MOI.ObjectiveSense())
    dest.c = zeros(size(dest.A, 2))
    if dest.sense != MOI.FEASIBILITY_SENSE
        obj = MOI.get(cache, MOI.ObjectiveFunction{SAF}())
        for term in obj.terms
            dest.c[term.variable.value] += term.coefficient
        end
        dest.constant = obj.constant
    end
    return index_map
end

# Index of the entry `(i, j)` with `i ≤ j` in the vectorization of the upper
# triangle
_triangle_index(i, j) = div((j - 1) * j, 2) + i

# Map the vectorization of the upper triangle of a symmetric matrix to the
# vectorization of the matrix used by `SDPSymmetryReduction.admissible_subspace`
function _square(v::AbstractVector, n)
    X = zeros(n, n)
    for j in 1:n, i in 1:j
        X[i, j] = X[j, i] = v[_triangle_index(i, j)]
    end
    return vec(X)
end

# Same as `_square` for each column of `A`, the result has one row per column
# of `A`
function _square(A::SparseArrays.SparseMatrixCSC, n)
    At = SparseArrays.sparse(A')
    I, J, V = Int[], Int[], Float64[]
    for j in 1:n, i in 1:j
        t = _triangle_index(i, j)
        for k in SparseArrays.nzrange(At, t)
            row = SparseArrays.rowvals(At)[k]
            val = SparseArrays.nonzeros(At)[k]
            push!(I, row)
            push!(J, i + (j - 1) * n)
            push!(V, val)
            if i != j
                push!(I, row)
                push!(J, j + (i - 1) * n)
                push!(V, val)
            end
        end
    end
    return SparseArrays.sparse(I, J, V, size(At, 1), n^2)
end

# Scalar product for `PositiveSemidefiniteConeTriangle` counts the
# off-diagonal entries twice so we scale them by `2` to use the standard
# scalar product
function _scale_off_diagonal(v::AbstractVector)
    n = MOI.Utilities.side_dimension_for_vectorized_dimension(length(v))
    s = Vector{Float64}(v)
    for j in 1:n, i in 1:(j-1)
        s[_triangle_index(i, j)] *= 2
    end
    return s
end

# Vectorization of the upper triangle of the block `B` as a sparse vector
function _triangle(B::AbstractMatrix)
    d = LinearAlgebra.checksquare(B)
    return SparseArrays.sparsevec([B[i, j] for j in 1:d for i in 1:j])
end

function _vector_affine_function(x, G::SparseArrays.SparseMatrixCSC, g)
    terms = MOI.VectorAffineTerm{Float64}[]
    for p in axes(G, 2), k in SparseArrays.nzrange(G, p)
        term = MOI.ScalarAffineTerm(SparseArrays.nonzeros(G)[k], x[p])
        push!(terms, MOI.VectorAffineTerm(SparseArrays.rowvals(G)[k], term))
    end
    return MOI.VectorAffineFunction(terms, Vector(g))
end

function _reduce(model::Optimizer)
    n = model.side_dimension
    # The dual of the problem is in the standard conic form
    #   min ⟨F₀, Z⟩ s.t. ⟨Fᵢ, Z⟩ = σ cᵢ, Z ⪰ 0
    # where `σ` is `1` for minimization and `-1` for maximization, which is
    # the form of `admissible_subspace`. The subspace is admissible for both
    # this problem and the original problem.
    σ = model.sense == MOI.MAX_SENSE ? -1.0 : 1.0
    P = SR.admissible_subspace(
        SR.Partition{UInt32},
        _square(model.a, n),
        _square(model.A, n),
        σ * model.c;
        verbose = model.verbose,
    )
    blkd = SR.blockDiagonalize(P, model.verbose)
    d = SR.dim(P)
    # `parts[t]` is the part of the `t`th entry of the upper triangle, the
    # entries of part `0` are zero
    parts = [Int(P.matrix[i, j]) for j in 1:n for i in 1:j]
    nonzero = findall(!iszero, parts)
    # `Π * z` is the vectorization of the matrix of the subspace with value
    # `z[p]` on the part `p`
    Π = SparseArrays.sparse(nonzero, parts[nonzero], 1.0, length(parts), d)
    # number of entries of the symmetric matrix in each part
    scaling = _scale_off_diagonal(ones(length(parts)))
    part_size = Π' * scaling
    # The orthogonal projection of the vectorization `v` of a matrix onto the
    # subspace is `Π * (W * v)`
    W = LinearAlgebra.Diagonal(1 ./ part_size) * Π' *
        LinearAlgebra.Diagonal(scaling)
    # The semidefinite constraint is replaced by its projection onto the
    # subspace `Π * (G * x + g)`.
    G, g = W * model.A, W * model.a
    # `blocks[k]` is the matrix such that the upper triangle of the `k`th block
    # of the matrix of the subspace with value `z[p]` on the part `p` is
    # `blocks[k] * z`
    blocks = [
        reduce(hcat, [_triangle(blkd.blks[p][k]) for p in 1:d]) for
        k in eachindex(blkd.blkSizes)
    ]
    psd_blocks = findall(>(1), blkd.blkSizes)
    scalar_blocks = findall(isone, blkd.blkSizes)
    # The columns of `G` may be linearly dependent, e.g., for the Lovász theta
    # number, the variables of the edges in the same orbit have the same
    # column. If the problem has a dual feasible solution, the objective is
    # zero in the kernel of `G` so we only keep the variables of a linearly
    # independent subset of its columns.
    F = LinearAlgebra.qr(G)
    columns = sort(F.pcol[1:LinearAlgebra.rank(F)])
    if model.verbose
        @info "Reduced problem has $(length(columns)) variables out of " *
              "$(size(G, 2)), semidefinite blocks of sizes " *
              "$(blkd.blkSizes[psd_blocks]) and $(length(scalar_blocks)) " *
              "blocks of size 1."
    end
    G = G[:, columns]
    reduced = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(reduced, length(columns))
    MOI.set(reduced, MOI.ObjectiveSense(), model.sense)
    if model.sense != MOI.FEASIBILITY_SENSE
        terms = MOI.ScalarAffineTerm.(model.c[columns], x)
        obj = MOI.ScalarAffineFunction(terms, model.constant)
        MOI.set(reduced, MOI.ObjectiveFunction{SAF}(), obj)
    end
    psd_cis = map(psd_blocks) do k
        func = _vector_affine_function(x, blocks[k] * G, blocks[k] * g)
        return MOI.add_constraint(reduced, func, PSD(blkd.blkSizes[k]))
    end
    scalar_ci = nothing
    scalar_block = reduce(
        vcat,
        blocks[scalar_blocks];
        init = SparseArrays.spzeros(0, d),
    )
    if !isempty(scalar_blocks)
        func = _vector_affine_function(x, scalar_block * G, scalar_block * g)
        set = MOI.Nonnegatives(length(scalar_blocks))
        scalar_ci = MOI.add_constraint(reduced, func, set)
    end
    return (;
        reduced,
        x,
        columns,
        Π,
        W,
        part_size,
        blocks,
        psd_blocks,
        psd_cis,
        scalar_block,
        scalar_ci,
    )
end

function MOI.optimize!(model::Optimizer)
    start = time()
    model.primal = model.dual = nothing
    r = _reduce(model)
    model.inner = MOI.instantiate(
        model.optimizer_constructor;
        with_cache_type = Float64,
        with_bridge_type = Float64,
    )
    if model.silent
        MOI.set(model.inner, MOI.Silent(), true)
    end
    index_map = MOI.copy_to(model.inner, r.reduced)
    MOI.optimize!(model.inner)
    primal_status = MOI.get(model.inner, MOI.PrimalStatus())
    if primal_status != MOI.NO_SOLUTION
        x = zeros(size(model.A, 2))
        x[r.columns] =
            MOI.get(model.inner, MOI.VariablePrimal(), [index_map[xi] for xi in r.x])
        a = _constant(model.a, primal_status)
        # `x` satisfies the semidefinite constraint projected onto the
        # subspace. As the subspace is admissible, there is a solution of the
        # original problem whose matrix `A * primal + a` is that projection.
        rhs = r.Π * (r.W * (model.A * x + a)) - a
        model.primal = LinearAlgebra.qr(model.A) \ rhs
        lhs = model.A * model.primal
        _check_residual(model, primal_status, "semidefinite constraint", lhs, rhs)
    end
    dual_status = MOI.get(model.inner, MOI.DualStatus())
    if dual_status != MOI.NO_SOLUTION
        # `w[p]` is the scalar product of the dual of the reduced cone
        # constraints with the image of the part `p`
        w = zeros(length(r.part_size))
        for (k, ci) in zip(r.psd_blocks, r.psd_cis)
            z = MOI.get(model.inner, MOI.ConstraintDual(), index_map[ci])
            w .+= r.blocks[k]' * _scale_off_diagonal(z)
        end
        if !isnothing(r.scalar_ci)
            z = MOI.get(model.inner, MOI.ConstraintDual(), index_map[r.scalar_ci])
            w .+= r.scalar_block' * z
        end
        model.dual = r.Π * (w ./ r.part_size)
        # The dual satisfies `A' * D * dual = σ * c` where `D` is diagonal
        # with entries `1` for diagonal entries and `2` for the off-diagonal
        # ones and `σ` is `1` for minimization, `-1` for maximization and `0`
        # for a feasibility problem or an infeasibility certificate.
        rhs = zeros(length(model.c))
        if dual_status != MOI.INFEASIBILITY_CERTIFICATE
            if model.sense == MOI.MAX_SENSE
                rhs .-= model.c
            elseif model.sense == MOI.MIN_SENSE
                rhs .+= model.c
            end
        end
        lhs = model.A' * _scale_off_diagonal(model.dual)
        _check_residual(model, dual_status, "dual equality constraints", lhs, rhs)
    end
    model.solve_time = time() - start
    return
end

# The constant of the semidefinite constraint is zero for an infeasibility
# certificate
function _constant(a, status)
    if status == MOI.INFEASIBILITY_CERTIFICATE
        return zero(a)
    end
    return a
end

# The residuals are only meaningful for a feasible point or an infeasibility
# certificate
function _check_residual(model::Optimizer, status, name, lhs, rhs)
    if status != MOI.FEASIBLE_POINT && status != MOI.INFEASIBILITY_CERTIFICATE
        return
    end
    residual = LinearAlgebra.norm(lhs - rhs, Inf)
    if residual > model.tol * (1 + LinearAlgebra.norm(rhs, Inf))
        @warn "The residual of the $name of the original problem is " *
              "$residual for the solution mapped back from the reduced problem."
    end
    return
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function MOI.get(model::Optimizer, attr::MOI.TerminationStatus)
    if isnothing(model.inner)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return MOI.get(model.inner, attr)
end

function MOI.get(model::Optimizer, attr::MOI.RawStatusString)
    if isnothing(model.inner)
        return "Optimize not called"
    end
    return MOI.get(model.inner, attr)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if isnothing(model.inner)
        return 0
    end
    return min(1, MOI.get(model.inner, MOI.ResultCount()))
end

function MOI.get(model::Optimizer, attr::Union{MOI.PrimalStatus,MOI.DualStatus})
    if attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return MOI.get(model.inner, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ObjectiveValue,MOI.DualObjectiveValue},
)
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model.inner, attr)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.primal[vi.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ::MOI.ConstraintIndex{VAF,PSD},
)
    MOI.check_result_index_bounds(model, attr)
    status = MOI.get(model, MOI.PrimalStatus())
    return model.A * model.primal + _constant(model.a, status)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ::MOI.ConstraintIndex{VAF,PSD},
)
    MOI.check_result_index_bounds(model, attr)
    return model.dual
end

end
