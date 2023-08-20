import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
import time

def opt_intersect(first: Zonotope, second: Zonotope):
    alpha = cp.Variable([first.g.shape[1]], name="alpha")
    beta = cp.Variable([second.g.shape[1]], name="beta")
    x = cp.Variable()

    obj = x

    constraints = [
        (alpha[k] <= 1) for k in range(first.g.shape[1])
    ]
    constraints += [
        (alpha[k] >= -1) for k in range(first.g.shape[1])
    ]
    constraints += [
        (beta[k] <= 1) for k in range(second.g.shape[1])
    ]
    constraints += [
        (beta[k] >= -1) for k in range(second.g.shape[1])
    ]
    constraints += [
        first.c + first.g @ alpha == second.c + second.g @ beta
    ]
    constraints += [
        x == 0
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    result = problem.solve()
    if np.isnan(result):
        return False
    else:
        return True

def reach_E(A_i_B_U):
    E = []
    for i in range(2):
        if i == 0:
            E.append(A_i_B_U[0])
        else:
            E.append(E[-1] + A_i_B_U[i])
    return E


A = np.array([[9.80246710e-01, 4.90147864e-01], [-9.80295727e-05, -4.90172374e-05]])
B = np.array([[1.97532898e-02], [9.80295727e-05]])

U = Zonotope.from_box(np.ones(1) * -10, np.ones(1) * 10)
A_i = [np.eye(A.shape[0])]
for i in range(5):
    A_i.append(A @ A_i[-1])
A_i_B_U = [val @ B @ U for val in A_i]
E = []
for i in range(4):
    if i == 0:
        E.append(A_i_B_U[0])
    else:
        E.append(E[-1] + A_i_B_U[i])
# after_reduce = E[2].order_reduction(2)

first = Zonotope.from_box(np.ones(2) * -0.02, np.ones(2) * 0.02)  # process noise
# second = A_i_B_U[0]

j = cp.Parameter(2, value=A_i_B_U[0].c)
h = cp.Parameter((2,1), value=A_i_B_U[0].g)
start = time.time()
alpha = np.ones([h.shape[0]])
beta = cp.Variable([h.shape[1]], name="beta")

constraints = [
    (beta[k] <= 1) for k in range(h.shape[1])
]
constraints += [
    (beta[k] >= -1) for k in range(h.shape[1])
]
constraints += [
    j + h @ beta <= first.c + first.g @ alpha
]
constraints += [
    j + h @ beta >= -1 * (first.c + first.g @ alpha)
]
problem = cp.Problem(cp.Minimize(0), constraints)
result = problem.solve()
end = time.time()
print(end - start)

# print(cp.installed_solvers())

# second = A_i_B_U[1]
j.value = A_i_B_U[1].c
h.value = A_i_B_U[1].g
start = time.time()
result = problem.solve(warm_start=True)
# opt_intersect(first, second)
end = time.time()
print(end - start)

# second = A_i_B_U[2]
j.value = A_i_B_U[2].c
h.value = A_i_B_U[2].g
start = time.time()
result = problem.solve(warm_start=True)
end = time.time()
print(end - start)
#
# second = A_i_B_U[3]
j.value = A_i_B_U[3].c
h.value = A_i_B_U[3].g
start = time.time()
result = problem.solve(warm_start=True)
end = time.time()
print(end - start)
#
# second = A_i_B_U[4]
j.value = A_i_B_U[4].c
h.value = A_i_B_U[4].g
start = time.time()
result = problem.solve(warm_start=True)
end = time.time()
print(end - start)


start = time.time()

end = time.time()
print(end - start)
exit()

# start = time.time()
# # opt_intersect(first, second)
# alpha = cp.Variable([first.g.shape[1]], name="alpha")
# beta = cp.Variable([second.g.shape[1]], name="beta")
# # x = cp.Variable()
#
# constraints = [
#     (alpha[k] <= 1) for k in range(first.g.shape[1])
# ]
# constraints += [
#     (alpha[k] >= -1) for k in range(first.g.shape[1])
# ]
# constraints += [
#     (beta[k] <= 1) for k in range(second.g.shape[1])
# ]
# constraints += [
#     (beta[k] >= -1) for k in range(second.g.shape[1])
# ]
# constraints += [
#     first.c + first.g @ alpha == second.c + second.g @ beta
# ]
# problem = cp.Problem(cp.Minimize(0), constraints)
# result = problem.solve()
# end = time.time()
# print(end - start)
#
# print(cp.installed_solvers())
#
# second = A_i_B_U[1]
# start = time.time()
# result = problem.solve()
# # opt_intersect(first, second)
# end = time.time()
# print(end - start)
#
# second = A_i_B_U[2]
# start = time.time()
# result = problem.solve()
# end = time.time()
# print(end - start)
# #
# second = A_i_B_U[3]
# start = time.time()
# result = problem.solve()
# end = time.time()
# print(end - start)
# #
# second = A_i_B_U[4]
# start = time.time()
# result = problem.solve()
# end = time.time()
# print(end - start)
