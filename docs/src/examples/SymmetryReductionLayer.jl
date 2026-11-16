# # Symmetry reduction layer for MathOptInterface
# Instead of reducing the problem by hand, the problem can be modeled with JuMP
# and reduced automatically by [`SDPSymmetryReduction.Optimizer`](@ref). It
# reduces the problem before giving it to the inner solver and maps the
# solution of the reduced problem back to the original problem.
#
# As an example, we compute the Lovász theta function of the Petersen graph
# ```math
# \vartheta(G)\coloneqq \inf\{t : tI + \sum_{ij \in E} y_{ij} (E_{ij} + E_{ji}) - J \succcurlyeq 0\}.
# ```

using JuMP
import CSDP
import LinearAlgebra
using SDPSymmetryReduction

edges = vcat(
    [(i, i % 5 + 1) for i in 1:5],
    [(i, i + 5) for i in 1:5],
    [(i + 5, (i + 1) % 5 + 6) for i in 1:5],
)
model = Model(() -> SDPSymmetryReduction.Optimizer(CSDP.Optimizer))
set_silent(model)
@variable(model, t)
@variable(model, y[edges])
@objective(model, Min, t)
S = t * Matrix(LinearAlgebra.I, 10, 10) - ones(10, 10)
for e in edges
    S[e[1], e[2]] += y[e]
    S[e[2], e[1]] += y[e]
end
@constraint(model, psd, LinearAlgebra.Symmetric(S) in PSDCone())
optimize!(model)
solution_summary(model)

# The 16 variables are reduced to 2 variables, as all the edges are in the same
# orbit, and the semidefinite constraint of size 10 is reduced to 3
# nonnegativity constraints. The optimal value is the Lovász theta function of
# the Petersen graph
objective_value(model)

# The primal and dual solutions are available as usual
value.(y)

# The dual of the semidefinite constraint is the solution of the dual problem
# ```math
# \sup\{\langle X,J\rangle : \langle X,I\rangle = 1, X_{ij} = 0 \text{ for } ij \in E, X\succcurlyeq 0\}.
# ```
dual(psd)
