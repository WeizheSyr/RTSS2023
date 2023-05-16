import os.path
import numpy as np
import cvxpy as cp
from simulators.linear.platoon import Platoon
import math
import time

# np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

INF = 1

max_index = 600
dt = 0.02
ref = [np.array([1])] * 301 + [np.array([1])] * 300
noise = {
    'process': {
        'type': 'white',
        'param': {'C': np.eye(7) * 0.01}
    }
}
platoon = Platoon('test', dt, max_index, noise)

A = platoon.sysd.A
B = platoon.sysd.B

inputsFilename = 'save/inputs_Platoontest.csv'
statesFilename = 'save/states_Platoontest.csv'
feedbacksFilename = 'save/feedbacks_Platoontest.csv'

if os.path.isfile(inputsFilename):
    with open(inputsFilename) as file:
        inputs = np.genfromtxt(file, delimiter=',')

if os.path.isfile(statesFilename):
    with open(statesFilename) as file:
        states = np.genfromtxt(file, delimiter=',')

if os.path.isfile(feedbacksFilename):
    with open(feedbacksFilename) as file:
        feedbacks = np.genfromtxt(file, delimiter=',')

# inputs = inputs.reshape(inputs.size, 1)

t = -90

# dimension of A
m = 7
# dimension of u
n = 4

# u_0, u_1, u_2, u_3
print("u_0, u_1 ... u_6")
u = np.zeros([m, n])
for i in range(m):
    u[i] = inputs[t+i]
print(u[0])

# x_0
print("x_0")
print(states[t])

print("y_0, y_1, ... y_6")
y = np.zeros([m, m])
for i in range(m):
    y[i] = feedbacks[t+i]
print(y[-1])

# with or without noise
delta = np.zeros([m, m])
for i in range(m):
    if i == 0:
        delta[0] = np.zeros(m)
    else:
        if i == 1:
            delta[1] = np.ones(m) * 0.01
        else:
            delta[i] = abs(A) @ delta[i-1] + delta[1]
for i in range(m):
    delta[i] += np.ones(m) * 0.01
print("delta")
print(delta)

A_k = np.zeros([m, m, m])
for i in range(m):
    if i == 0:
        A_k[0] = np.eye(m)
    else:
        A_k[i] = A @ A_k[i-1]
print("A_k")
print(A_k[1])

# print(states[1])

U = np.zeros([m, m])
for i in range(m):
    if i == 0:
        U[0] = np.zeros([m])
    else:
        U[i] = B @ u[i-1] + A @ U[i-1]
print("U")
print(U)

x = cp.Variable([m], name="x")
gama = cp.Variable([m], name="gama", boolean=True)
E = cp.Variable([m, m], name="E")

obj = cp.sum(gama)

constraints = [
    (y[k] - U[k] - (A_k[k] @ x) - E[k] <= delta[k]) for k in range(m)
]
constraints += [
    (y[k] - U[k] - (A_k[k] @ x) - E[k] >= -delta[k]) for k in range(m)
]

constraints += [
    (E[:, k] <= INF * gama[k] * np.ones(m)) for k in range(m)
]

constraints += [
    (E[:, k] >= -1 * INF * gama[k] * np.ones(m)) for k in range(m)
]

problem = cp.Problem(cp.Minimize(obj), constraints)

start = time.perf_counter()
problem.solve()
# problem.solve(solver=cp.SCIPY)
end = time.perf_counter()
elapsed = end - start
print(elapsed * 1000, "ms")

# Print result.
print("The optimal value is", problem.value)
print("A solution X is")
print(x.value)
print("A solution gama is")
print(gama.value)
print("the real state is")
print(states[t])
print("y_0 is")
print(y[0])
print("E.value")
print(E.value)


# solve the bound of the second dimension
x1 = cp.Variable([m], name="x1")
y1 = cp.Variable([m, m], name="y1")
alpha = cp.Variable([m], name="alpha", boolean=True)
beta = cp.Variable([m], name="beta", boolean=True)

obj1 = cp.abs(x1[1] - x.value[1])

x_ = np.zeros([m])
for i in range(m):
    x_[i] = x.value[i]
E_ = np.zeros([m, m])
for i in range(m):
    for j in range(m):
        E_[i][j] = E.value[i][j]

print("A_k[i] @ x_", A_k[i] @ x_)
# temp = np.zeros([m, m])
# for i in range(m):
#     temp[i] = A_k[i] @ x_ - A_k[i] @ x1

temp1 = np.zeros([m, m])
for i in range(m):
    temp1[i] = A_k[i] @ x_

sigma = y - temp1 - E_

constraints1 = [
    cp.sum(beta) <= m
]
constraints1 += [
    ((((y - y1) - (A_k[k] @ x_ - A_k[i] @ x1) - sigma)[:, k] <= beta[k] * np.ones(m)) for i in range(m)) for k in range(m)
]
constraints1 += [
    ((((y - y1) - (A_k[k] @ x_ - A_k[i] @ x1) - sigma)[:, k] >= -beta[k] * np.ones(m)) for i in range(m)) for k in range(m)
]
constraints1 += [
    y - y1 - sigma <= 2 * delta
]
constraints1 += [
    y - y1 - sigma >= -2 * delta
]

bound = cp.Problem(cp.Maximize(obj1), constraints1)
bound.solve()
print("The optimal value is", bound.value)