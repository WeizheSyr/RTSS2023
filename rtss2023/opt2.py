import os.path
import numpy as np
import cvxpy as cp
from simulators.linear.platoon import Platoon
import math

np.set_printoptions(precision=6)
np.set_printoptions(suppress = True)

INF = 100000
# INF = math.inf

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

t = -80

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

delta = np.zeros([m, m])
for i in range(m):
    if i == 0:
        delta[0] = np.ones(m) * 0
    else:
        if i == 1:
            delta[1] = np.ones(m) * 0.01
        else:
            delta[i] = abs(A) @ delta[i-1] + delta[1]
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
        U[i] = B @ u[i]
print("U")
print(U[-1])

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
    (E[:, k] <= INF * gama[k] * np.ones(m).T) for k in range(m)
]

constraints += [
    (E[:, k] >= INF * -gama[k] * np.ones(m).T) for k in range(m)
]

problem = cp.Problem(cp.Minimize(obj), constraints)

problem.solve()

# Print result.
print("The optimal value is", problem.value)
print("A solution X is")
print(x.value)
print(gama.value)
# print(E.value)

print(states[t])
print("********************")
print(A @ states[t-1] + B @ inputs[t-1])
# print("*******************#############*")
# print(inputs[t])
print("********************")
