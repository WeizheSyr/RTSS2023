import os.path
import numpy as np
import cvxpy as cp

np.set_printoptions(precision=5)

cf_ = 155494.663
cr_ = 155494.663
wheelbase_ = 2.852
steer_ratio_ = 16
steer_single_direction_max_degree_ = 470.0
mass_fl = 520
mass_fr = 520
mass_rl = 520
mass_rr = 520
mass_front = mass_fl + mass_fr
mass_rear = mass_rl + mass_rr
mass_ = mass_front + mass_rear
lf_ = wheelbase_ * (1.0 - mass_front / mass_)
lr_ = wheelbase_ * (1.0 - mass_rear / mass_)
iz_ = lf_ * lf_ * mass_front + lr_ * lr_ * mass_rear
vx = 5

A = np.zeros((4, 4), dtype=np.float32)
A[0, 1] = 1.0
A[1, 1] = -(cf_ + cr_) / mass_ / vx
A[1, 2] = (cf_ + cr_) / mass_
A[1, 3] = (lr_ * cr_ - lf_ * cf_) / mass_ / vx
A[2, 3] = 1.0
A[3, 1] = (lr_ * cr_ - lf_ * cf_) / iz_ / vx
A[3, 2] = (lf_ * cf_ - lr_ * cr_) / iz_
A[3, 3] = -1.0 * (lf_ * lf_ * cf_ + lr_ * lr_ * cr_) / iz_ / vx

B = np.zeros((4, 1), dtype=np.float32)
B[1, 0] = cf_ / mass_
B[3, 0] = lf_ * cf_ / iz_

inputsFilename = 'save/inputs_Lane Keeping test.csv'
statesFilename = 'save/states_Lane Keeping test.csv'
feedbacksFilename = 'save/feedbacks_Lane Keeping test.csv'

if os.path.isfile(inputsFilename):
    with open(inputsFilename) as file:
        inputs = np.genfromtxt(file, delimiter=',')

if os.path.isfile(statesFilename):
    with open(statesFilename) as file:
        states = np.genfromtxt(file, delimiter=',')

if os.path.isfile(feedbacksFilename):
    with open(feedbacksFilename) as file:
        feedbacks = np.genfromtxt(file, delimiter=',')

inputs = inputs.reshape(inputs.size, 1)

t = -40

# u_0, u_1, u_2, u_3
print("u_0, u_1, u_2, u_3")
u = np.empty([4])
for i in range(4):
    u[i] = inputs[t+i]
print(u)

# x_0
print("x_0")
print(states[t])

print("y_0, y_1, y_2, y_3")
y = np.empty([4, 4])
for i in range(4):
    y[i] = feedbacks[t+i]
print(y)

delta = np.empty([4, 4])
delta[0] = np.empty(4)
delta[1] = np.ones(4) * 0.001
delta[2] = abs(A) @ np.ones(4) * 0.001 + delta[1]
delta[3] = abs(A) @ abs(A) @ np.ones(4) * 0.001 + delta[2]


print("@@@@@@@@@@@@@@@@@@@@@@@@@")
print(states[t])


A_k = np.empty([4, 4, 4])
A_k[0] = np.eye(4)
A_k[1] = A @ A_k[0]
A_k[2] = A @ A
A_k[3] = A @ A @ A

U = np.empty([4, 4, 1])
u = u.reshape(4, 1)
U[0] = np.empty([4, 1])
U[1] = B @ u[0].reshape(1, 1)
U[2] = A @ B @ u[0].reshape(1, 1) + B @ u[1].reshape(1, 1)
U[3] = A_k[2] @ B @ u[0].reshape(1, 1) + A @ B @ u[1].reshape(1, 1) + B @ u[2].reshape(1, 1)
print(U[1].shape)
print(y[1].shape)

U1 = np.empty([4, 4])
for i in range(4):
    for j in range(4):
        U1[i][j] = U[i][j][0]
print("U1")
print(U1[3])
print(U[3])

x = cp.Variable(4, name="x")
gama = cp.Variable(4, name="gama", boolean=True)
# gama = cp.Variable(4, name="gama")
E = cp.Variable([4, 4], name="E")

obj = gama[0] + gama[1] + gama[2] + gama[3]

constraints = [
    (y[k] - U1[k] - A_k[k] @ x - E[k] <= delta[k]) for k in range(4)
]
constraints += [
    (y[k] - U1[k] - A_k[k] @ x - E[k] >= -delta[k]) for k in range(4)
]

constraints += [
    (E[:, k] <= 10 * gama[k] * np.ones(4).T) for k in range(4)
]

constraints += [
    (E[:, k] >= 10 * -gama[k] * np.ones(4).T) for k in range(4)
]

problem = cp.Problem(cp.Minimize(obj), constraints)

problem.solve()

# Print result.
print("The optimal value is", problem.value)
print("A solution X is")
print(x.value)

print(states[t])

