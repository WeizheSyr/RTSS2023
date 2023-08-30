import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
from utils.Baseline import Platoon
import time

A = Platoon.model.sysd.A
B = Platoon.model.sysd.B

U = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)
A_i = [np.eye(A.shape[0])]
for i in range(20):
    A_i.append(A @ A_i[-1])
A_i_B_U = [val @ B @ U for val in A_i]
E = []
for i in range(20):
    if i == 0:
        E.append(A_i_B_U[0])
    else:
        E.append(E[-1] + A_i_B_U[i])
        # E.append(A_i_B_U[0])

temp = E[9].order_reduction(4)
print(temp)
exit()

d = 10
# authTimestep + self.auth.timestep
k = 9

# A^d
A_d = [np.eye(A.shape[0])]
for i in range(20):
    A_d.append(A @ A_d[-1])

# I + A^(k-1)
A_k_1 = []
for i in range(k):
    if i == 0:
        A_k_1.append(np.eye(A.shape[0]))
    else:
        A_k_1.append(A_k_1[-1] + A_d[i - 1])

# F()
F = A_d[d - 1] @ A_k_1[k - 1]

# Parameters
D_low = cp.Parameter(A.shape[0])
D_up = cp.Parameter(A.shape[0])

# D_low.value = [-0.22060118825057262, -0.2201856099204006, -0.028929731712184048, -1.1590693513020622, -0.8727438691416309, -0.7056735968203992, -0.43328699395346926]
# D_up.value = [-0.4023044399373612, -0.3933730907986046, -0.19501647114507392, -1.316595325577472, -1.0624287153104497, -0.8946908257562018, -0.5895528329640749]

D_low.value = [-10, -10, -10, -10, -10, -10, -10]
D_up.value = [10, 10, 10, 10, 10, 10, 10]
# Variables
tau = cp.Variable(7, name="delta_tau")
alpha = cp.Variable([E[d-1].g.shape[1]], name="alpha")

# Constraints
constraints = [
    ((temp.c + temp.g @ alpha)[f] <= D_up[f] + (F @ tau)[f] / 2 + cp.norm1((F @ tau)[f] / 4)) for f in range(A.shape[0])
]
constraints += [
    ((temp.c + temp.g @ alpha)[f] >= D_low[f] - (F @ tau)[f] / 2 - cp.norm1((F @ tau)[f] / 2)) for f in range(A.shape[0])
]
constraints += [
    (alpha[f] <= 1) for f in range(4)
]
constraints += [
    (alpha[f] >= -1) for f in range(4)
]
# constraints += [
#     (tau[f] <= 0) for f in range(A.shape[0])
# ]

obj = cp.Minimize(cp.norm2(tau))
problem = cp.Problem(obj, constraints)
start = time.time()
# problem = cp.Problem(cp.Minimize(0), constraints)
result = problem.solve()
end = time.time()

print(tau.value)
print(end - start)
