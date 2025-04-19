import casadi as ca
import time

# Define penalty weight
lambda_penalty = 1000  # Stronger penalty weight

# Define optimization problems with equivalent formulations
opti_barrier = ca.Opti()
var_x_barrier = opti_barrier.variable()
opti_barrier.subject_to(
    var_x_barrier + 0.2 > 0
)  # Ensures feasibility explicitly
opti_barrier.minimize(var_x_barrier - ca.log(var_x_barrier + 0.2))
opti_barrier.set_initial(var_x_barrier, 0.5)
solver_barrier = opti_barrier.solver("ipopt")

opti_penalty = ca.Opti()
var_x_penalty = opti_penalty.variable()
c_x_penalty = var_x_penalty + 0.2  # Compute constraint function within Opti
reciprocal_penalty = (
    lambda_penalty * (1 / c_x_penalty - 1) ** 2
)  # Corrected penalty term
opti_penalty.minimize(var_x_penalty + reciprocal_penalty)
opti_penalty.subject_to(
    var_x_penalty + 0.2 > 0
)  # Enforce feasibility explicitly
opti_penalty.set_initial(var_x_penalty, 0.5)
solver_penalty = opti_penalty.solver("ipopt")

# Solve the barrier function optimization
start_time = time.time()
sol_barrier = opti_barrier.solve()
time_barrier = time.time() - start_time

# Solve the soft constraint (penalty) optimization
start_time = time.time()
sol_penalty = opti_penalty.solve()
time_penalty = time.time() - start_time

# Display the solutions and times
print("Barrier Function Optimization:")
print("Optimal x:", sol_barrier.value(var_x_barrier))
print("Time taken:", time_barrier, "seconds")

print("\nSoft Constraint (Penalty) Optimization:")
print("Optimal x:", sol_penalty.value(var_x_penalty))
print("Time taken:", time_penalty, "seconds")
