import os.path
import numpy as np
import cvxpy as cp
from simulators.linear.quadrotor import Quadrotor
import math
import time

# np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

INF = 0.5

max_index = 600
dt = 0.02
ref = [np.array([1])] * 301 + [np.array([1])] * 400
noise = {
    'process': {
        'type': 'box_uniform',
        'param': {'lo': np.ones(12) * -0.001,
                  'up': np.ones(12) * 0.001}
    },
    'measurement': {
        'type': 'box_uniform',
        'param': {'lo': np.ones(6) * -0.001,
                  'up': np.ones(6) * 0.001}
    }
}
quadrotor = Quadrotor('test', dt, max_index, noise)

A = quadrotor.sysd.A
B = quadrotor.sysd.B

inputsFilename = 'save/inputs_Quadrotor test.csv'
statesFilename = 'save/states_Quadrotor test.csv'
feedbacksFilename = 'save/feedbacks_Quadrotor test.csv'

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

# print(inputs[100])

t = -70

# dimension of A
m = 12
# dimension of u
n = 12
# dimension of timestep
timestep = 12

# u_0, u_1, u_2, u_3
# print("u_0, u_1 ... u_6")
u = np.zeros([timestep, n])
for i in range(timestep):
    u[i] = inputs[t + i]
# print(u[0])

# x_0
print("x_0")
print(states[t])

# print("y_0, y_1, ... y_6")
y = np.zeros([timestep, m])
for i in range(timestep):
    y[i] = feedbacks[t + i]
# print(y[-1])

# with or without noise
delta = np.zeros([timestep, m])
for i in range(timestep):
    if i == 0:
        delta[0] = np.zeros(m)
    else:
        if i == 1:
            delta[1] = np.ones(m) * 0.01
        else:
            delta[i] = abs(A) @ delta[i - 1] + delta[1]
for i in range(m):
    delta[i] += np.ones(m) * 0.01
# print("delta")
# print(delta)

A_k = np.zeros([timestep, m, m])
for i in range(timestep):
    if i == 0:
        A_k[0] = np.eye(m)
    else:
        A_k[i] = A @ A_k[i - 1]
# print("A_k")
# print(A_k[1])

# print(states[1])

U = np.zeros([timestep, m])
for i in range(timestep):
    if i == 0:
        U[0] = np.zeros([m])
    else:
        print("########")
        # print(u[i-1].reshape(-1,1).shape)
        # print(U[i-1].reshape(-1,1).shape)
        # print(B.shape)
        # print(A.shape)
        U[i] = B @ u[i - 1] + A @ U[i - 1]
# print("U")
# print(U)

x = cp.Variable([m], name="x")
gama = cp.Variable([m], name="gama", boolean=True)
E = cp.Variable([timestep, m], name="E")

obj = cp.sum(gama)

constraints = [
    (gama[k] <= 0) for k in range(m)
]
att = 2
del constraints[att]
constraints += [
    gama[att] <= 1
]
constraints += [
    (y[k] - U[k] - (A_k[k] @ x) - E[k] <= delta[k]) for k in range(timestep)
]
constraints += [
    (y[k] - U[k] - (A_k[k] @ x) - E[k] >= -delta[k]) for k in range(timestep)
]

constraints += [
    (E[:, k] <= INF * gama[k] * np.ones(timestep)) for k in range(m)
]

constraints += [
    (E[:, k] >= -1 * INF * gama[k] * np.ones(timestep)) for k in range(m)
]

problem = cp.Problem(cp.Minimize(obj), constraints)

# start = time.perf_counter()
# problem.solve()
problem.solve(solver=cp.SCIPY)
# end = time.perf_counter()
# elapsed = end - start
# print(elapsed * 1000, "ms")

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
# print("E.value")
# print(E.value)




# stealth attack bound2
x1 = cp.Variable([m], name="x1")
E1 = cp.Variable([timestep, m], name="E1")
beta = cp.Variable([m], name="beta", boolean=True)

dim = 0
obj1 = x1[dim] - x.value[dim]
# obj1 = x1[dim] - x.value

x_ = np.zeros([m])
for i in range(m):
    x_[i] = x.value[i]
E_ = np.zeros([timestep, m])
for i in range(timestep):
    for j in range(m):
        E_[i][j] = E.value[i][j]

# print("A_k[i] @ x_", A_k[i] @ x_)

temp = cp.Variable([timestep, m])
# temp = np.zeros([m, m])
# for i in range(m):
#     temp[i] = A_k[i] @ (x_ - x1)

temp1 = np.zeros([timestep, m])
for i in range(timestep):
    temp1[i] = A_k[i] @ x_

# sigma = y - temp1 - E_

# constraints1 = [
#     (beta[k] <= 1 + gama[k].value) for k in range(m)
# ]
constraints1 = [
    cp.sum(beta) <= 1
]
# del constraints1[att]
# constraints1 += [
#     beta[att] <= 1
# ]
constraints1 += [
    (y[k] - U[k] - (A_k[k] @ x1) - E1[k] <= delta[k]) for k in range(timestep)
]
constraints1 += [
    (y[k] - U[k] - (A_k[k] @ x1) - E1[k] >= -delta[k]) for k in range(timestep)
]

constraints1 += [
    (E1[:, k] <= INF * beta[k] * np.ones(timestep)) for k in range(m)
]
constraints1 += [
    (E1[:, k] >= -1 * INF * beta[k] * np.ones(timestep)) for k in range(m)
]
bound1 = cp.Problem(cp.Maximize(obj1), constraints1)
bound2 = cp.Problem(cp.Minimize(obj1), constraints1)
bound1.solve()
bound2.solve()
print("The optimal value is", bound1.value, bound2.value)

# path = 'save/each_dim.txt'
# with open(path, 'a') as target:
#     # target.writelines(["\nattack_dim: ", str(att), "\n"])
#     # target.writelines([str(x.value), "\n"])
#     # target.writelines([str(states[t]), "\n"])
#     target.writelines(["dim: ", str(dim), "\n"])
#     target.writelines([str(bound1.value), " ", str(bound2.value), "\n"])