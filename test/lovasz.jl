@testset "Lovász ϑ′ for Erdös-Renyi graphs" begin
    dim = SDPSymmetryReduction.dim
    @testset "PGL(2,q=3)" begin
        CAb = Lovászϑ′_ER_graph(3)
        P = SDPSymmetryReduction.admissible_subspace(CAb...)
        @test SDPSymmetryReduction.dim(P) == 12
        Q_hat = SDPSymmetryReduction.diagonalize(Float64, P)
        @test sort(size.(Q_hat, 2)) == [2, 2, 3]

        model = opt_model(P, Q_hat, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 5.0 rtol = 1e-7
    end

    @testset "PGL(2,q=5)" begin
        CAb = Lovászϑ′_ER_graph(5)
        P = SDPSymmetryReduction.admissible_subspace(CAb...)
        @test SDPSymmetryReduction.dim(P) == 15
        Q_hat = SDPSymmetryReduction.diagonalize(Float64, P)
        @test sort(size.(Q_hat, 2)) == [2, 2, 2, 3]

        model = opt_model(P, Q_hat, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 10.066926 rtol = 1e-7
    end

    @testset "PGL(2,q=5)" begin
        CAb = Lovászϑ′_ER_graph(7)
        P = SDPSymmetryReduction.admissible_subspace(CAb...)
        @test SDPSymmetryReduction.dim(P) == 18
        Q_hat = SDPSymmetryReduction.diagonalize(Float64, P)
        @test sort(size.(Q_hat, 2)) == [2, 2, 2, 2, 3]

        model = opt_model(P, Q_hat, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 15.743402 rtol = 1e-7
    end
end
