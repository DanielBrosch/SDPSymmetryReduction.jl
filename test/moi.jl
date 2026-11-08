using Test
using SDPSymmetryReduction
import CSDP
import JuMP
import LinearAlgebra
import Logging
import MathOptInterface as MOI

const PSD = MOI.PositiveSemidefiniteConeTriangle

# The Lovász theta number is
#   min t s.t. t I + ∑_{ij ∈ E} y_ij (Eᵢⱼ + Eⱼᵢ) - J ⪰ 0
# The dual of the semidefinite constraint is the solution `X` of
#   max ⟨J, X⟩ s.t. tr(X) = 1, X_ij = 0 for ij ∈ E, X ⪰ 0
function lovasz_theta(optimizer, n, edges)
    model = JuMP.Model(optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, t)
    JuMP.@variable(model, y[edges])
    JuMP.@objective(model, Min, t)
    S = t * Matrix(LinearAlgebra.I, n, n) - ones(n, n)
    for e in edges
        S[e[1], e[2]] += y[e]
        S[e[2], e[1]] += y[e]
    end
    JuMP.@constraint(model, psd, LinearAlgebra.Symmetric(S) in JuMP.PSDCone())
    JuMP.optimize!(model)
    return model
end

cycle_edges(n) = [(i, i % n + 1) for i in 1:n]

function petersen_edges()
    outer = cycle_edges(5)
    spokes = [(i, i + 5) for i in 1:5]
    inner = [(i + 5, (i + 1) % 5 + 6) for i in 1:5]
    return vcat(outer, spokes, inner)
end

# Polarity graph of the projective plane over the finite field with `q` elements
function erdos_renyi_edges(q)
    PG2q = vcat(
        [[0, 0, 1]],
        [[0, 1, b] for b in 0:q-1],
        [[1, a, b] for a in 0:q-1 for b in 0:q-1],
    )
    n = length(PG2q)
    return [(i, j) for j in 1:n for i in 1:(j-1) if PG2q[i]' * PG2q[j] % q == 0]
end

# The optimal solutions may not be unique so we only compare them with the
# solution of the problem that is not reduced up to `atol`
function test_lovasz_theta(n, edges, expected = nothing; atol = 1e-6)
    # Test that the residuals of the solution mapped back are small
    reduced = @test_logs min_level = Logging.Warn lovasz_theta(
        () -> SDPSymmetryReduction.Optimizer(CSDP.Optimizer),
        n,
        edges,
    )
    full = lovasz_theta(CSDP.Optimizer, n, edges)
    if isnothing(expected)
        expected = JuMP.objective_value(full)
    end
    @test JuMP.solver_name(reduced) == "SDPSymmetryReduction with CSDP"
    @test JuMP.termination_status(reduced) == JuMP.OPTIMAL
    @test JuMP.primal_status(reduced) == JuMP.FEASIBLE_POINT
    @test JuMP.dual_status(reduced) == JuMP.FEASIBLE_POINT
    @test JuMP.result_count(reduced) == 1
    @test JuMP.solve_time(reduced) > 0
    @test JuMP.objective_value(reduced) ≈ expected rtol = 1e-6
    @test JuMP.dual_objective_value(reduced) ≈ expected rtol = 1e-6
    @test JuMP.value(reduced[:t]) ≈ expected rtol = 1e-6
    y = [JuMP.value(reduced[:y][e]) for e in edges]
    @test y ≈ [JuMP.value(full[:y][e]) for e in edges] atol = atol
    S = JuMP.value(reduced[:psd])
    @test S ≈ JuMP.value(full[:psd]) atol = atol
    @test LinearAlgebra.eigmin(S) > -1e-6
    X = JuMP.dual(reduced[:psd])
    @test X ≈ JuMP.dual(full[:psd]) atol = atol
    @test LinearAlgebra.eigmin(X) > -1e-6
    @test LinearAlgebra.tr(X) ≈ 1 rtol = 1e-6
    @test sum(X) ≈ expected rtol = 1e-6
    @test LinearAlgebra.dot(X, S) ≈ 0 atol = 1e-6
    return
end

# `I - y J ⪰ 0` with the objective `y - 1` maximized or `1 - y` minimized,
# the solution is `y = 1 / n` and the dual is `J / n²`
function test_scaled_identity(sense)
    n = 4
    src = MOI.Utilities.Model{Float64}()
    y = MOI.add_variable(src)
    MOI.set(src, MOI.ObjectiveSense(), sense)
    σ = sense == MOI.MAX_SENSE ? 1.0 : -1.0
    obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(σ, y)], -σ)
    MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    identity = [i == j ? 1.0 : 0.0 for j in 1:n for i in 1:j]
    terms = [
        MOI.VectorAffineTerm(t, MOI.ScalarAffineTerm(-1.0, y)) for
        t in eachindex(identity)
    ]
    func = MOI.VectorAffineFunction(terms, identity)
    c = MOI.add_constraint(src, func, PSD(n))
    model = SDPSymmetryReduction.Optimizer(CSDP.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    @test MOI.get(model, MOI.Silent())
    index_map = MOI.copy_to(model, src)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ σ * (1 / n - 1) rtol = 1e-6
    @test MOI.get(model, MOI.VariablePrimal(), index_map[y]) ≈ 1 / n rtol = 1e-6
    S = [i == j ? 1 - 1 / n : -1 / n for j in 1:n for i in 1:j]
    @test MOI.get(model, MOI.ConstraintPrimal(), index_map[c]) ≈ S atol = 1e-6
    X = fill(1 / n^2, length(S))
    @test MOI.get(model, MOI.ConstraintDual(), index_map[c]) ≈ X atol = 1e-6
    return
end

function psd_function(x)
    term = MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x))
    return MOI.VectorAffineFunction([term], zeros(3))
end

@testset "MathOptInterface layer" begin
    @testset "Lovász ϑ of the cycle C₅" begin
        test_lovasz_theta(5, cycle_edges(5), √5)
    end
    @testset "Lovász ϑ of the Petersen graph" begin
        test_lovasz_theta(10, petersen_edges(), 4)
    end
    @testset "Lovász ϑ of the Erdös-Renyi graph for q = 3" begin
        # The reduced problem has semidefinite blocks of size 2 and 3
        test_lovasz_theta(13, erdos_renyi_edges(3); atol = 1e-2)
    end
    @testset "$sense" for sense in [MOI.MIN_SENSE, MOI.MAX_SENSE]
        test_scaled_identity(sense)
    end
    @testset "Infeasible" begin
        # `[-1 x; x -1] ⪰ 0` is infeasible
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        term = MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x))
        func = MOI.VectorAffineFunction([term], [-1.0, 0.0, -1.0])
        c = MOI.add_constraint(src, func, PSD(2))
        model = SDPSymmetryReduction.Optimizer(CSDP.Optimizer)
        MOI.set(model, MOI.Silent(), true)
        index_map = MOI.copy_to(model, src)
        @test_logs min_level = Logging.Warn MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(model, MOI.RawStatusString()) isa String
        @test MOI.get(model, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        # `Z = s I` with `s > 0`
        Z = MOI.get(model, MOI.ConstraintDual(), index_map[c])
        @test Z[1] > 0
        @test Z ≈ [Z[1], 0, Z[1]] atol = 1e-6
    end
    @testset "Unbounded" begin
        # `min -x s.t. [x 1; 1 x] ⪰ 0` is unbounded
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        obj = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-1.0, x)], 0.0)
        MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
        terms = [
            MOI.VectorAffineTerm(t, MOI.ScalarAffineTerm(1.0, x)) for
            t in [1, 3]
        ]
        func = MOI.VectorAffineFunction(terms, [0.0, 1.0, 0.0])
        c = MOI.add_constraint(src, func, PSD(2))
        model = SDPSymmetryReduction.Optimizer(CSDP.Optimizer)
        MOI.set(model, MOI.Silent(), true)
        index_map = MOI.copy_to(model, src)
        @test_logs min_level = Logging.Warn MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        # The ray has no constant
        r = MOI.get(model, MOI.VariablePrimal(), index_map[x])
        @test r > 0
        @test MOI.get(model, MOI.ConstraintPrimal(), index_map[c]) ≈ [r, 0, r] atol = 1e-6
    end
    @testset "Residual warnings" begin
        # `min x s.t. [x 1; 1 x] ⪰ 0`
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
        terms = [
            MOI.VectorAffineTerm(t, MOI.ScalarAffineTerm(1.0, x)) for
            t in [1, 3]
        ]
        func = MOI.VectorAffineFunction(terms, [0.0, 1.0, 0.0])
        MOI.add_constraint(src, func, PSD(2))
        # A negative tolerance forces the warnings
        model = SDPSymmetryReduction.Optimizer(
            CSDP.Optimizer;
            verbose = true,
            tol = -1,
        )
        MOI.set(model, MOI.Silent(), true)
        MOI.copy_to(model, src)
        @test_logs(
            (:info, r"^Reduced problem has"),
            (:warn, r"semidefinite constraint"),
            (:warn, r"dual equality constraints"),
            match_mode = :any,
            MOI.optimize!(model),
        )
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1 rtol = 1e-6
    end
    @testset "Unsupported" begin
        model = SDPSymmetryReduction.Optimizer(CSDP.Optimizer)
        @test MOI.is_empty(model)
        @test MOI.get(model, MOI.SolverName()) == "SDPSymmetryReduction"
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(model, MOI.RawStatusString()) == "Optimize not called"
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        MOI.add_constraint(src, x, MOI.GreaterThan(0.0))
        err = MOI.UnsupportedConstraint{MOI.VariableIndex,MOI.GreaterThan{Float64}}
        @test_throws err MOI.copy_to(model, src)
        # The standard conic form is not supported
        src = MOI.Utilities.Model{Float64}()
        MOI.add_constrained_variables(src, PSD(2))
        err = MOI.UnsupportedConstraint{MOI.VectorOfVariables,PSD}
        @test_throws err MOI.copy_to(model, src)
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        MOI.add_constraint(src, psd_function(x), PSD(2))
        MOI.add_constraint(src, psd_function(x), PSD(2))
        @test_throws ErrorException MOI.copy_to(model, src)
        src = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        x = MOI.add_variable(src)
        MOI.add_constraint(src, psd_function(x), PSD(2))
        MOI.set(src, MOI.VariablePrimalStart(), x, 1.0)
        err = MOI.UnsupportedAttribute{MOI.VariablePrimalStart}
        @test_throws err MOI.copy_to(model, src)
        src = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        x = MOI.add_variable(src)
        c = MOI.add_constraint(src, psd_function(x), PSD(2))
        MOI.set(src, MOI.ConstraintPrimalStart(), c, ones(3))
        err = MOI.UnsupportedAttribute{MOI.ConstraintPrimalStart}
        @test_throws err MOI.copy_to(model, src)
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(src)
        MOI.add_constraint(src, psd_function(x), PSD(2))
        obj = 1.0 * x * x
        MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
        err = MOI.UnsupportedAttribute{MOI.ObjectiveFunction{typeof(obj)}}
        @test_throws err MOI.copy_to(model, src)
    end
end
