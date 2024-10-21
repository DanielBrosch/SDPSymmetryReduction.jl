@testset "Lovász ϑ′ for Erdös-Renyi graphs" begin
    @testset "PGL(2,q=3)" begin
        CAb = Lovászϑ′_ER_graph(3)
        P = admPartSubspace(CAb..., false)
        blocks = blockDiagonalize(P, false)
        model = opt_model(P, blocks, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test admPartSubspace(CAb..., false).n == 12
        @test sort(blocks.blkSizes) == [2, 2, 3]
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 5.0 rtol = 1e-7
    end

    @testset "PGL(2,q=5)" begin
        CAb = Lovászϑ′_ER_graph(5)
        P = admPartSubspace(CAb..., false)
        blocks = blockDiagonalize(P, false)
        model = opt_model(P, blocks, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test admPartSubspace(CAb..., false).n == 15
        @test sort(blocks.blkSizes) == [2, 2, 2, 3]
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 10.066926 rtol = 1e-7
    end

    @testset "PGL(2,q=5)" begin
        CAb = Lovászϑ′_ER_graph(7)
        P = admPartSubspace(CAb..., false)
        blocks = blockDiagonalize(P, false)
        model = opt_model(P, blocks, CAb)
        JuMP.set_optimizer(model, CSDP.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)

        @test admPartSubspace(CAb..., false).n == 18
        @test sort(blocks.blkSizes) == [2, 2, 2, 2, 3]
        @test JuMP.termination_status(model) == JuMP.OPTIMAL
        @test JuMP.objective_value(model) ≈ 15.743402 rtol = 1e-7
    end
end
