import pulp as pl
import os.path
import numpy as np

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

t = -50

# u_0, u_1, u_2, u_3
print("u_0, u_1, u_2, u_3")
u = np.empty([4])
for i in range(4):
    u[i] = inputs[t+i]
print(u)

# x_0
print("x_0")
print(states[-50])

print("y_0, y_1, y_2, y_3")
y = np.empty([4,4])
for i in range(4):
    y[i] = feedbacks[t+i]
print(y)
print(type(y[1]))

delta = np.empty([4,4])
delta[0] = np.ones(4) * 0.001
delta[1] = abs(A) @ np.ones(4) * 0.001 + delta[0]
delta[2] = abs(A) @ abs(A) @ np.ones(4) * 0.001 + delta[1]
delta[3] = abs(A) @ abs(A) @ abs(A) @ np.ones(4) * 0.001 + delta[2]

# create the model
model = pl.LpProblem(name="test", sense=pl.LpMinimize)

# Initialize the decision variables
x = pl.LpVariable.matrix("x", range(4), cat=pl.LpContinuous)
gama = pl.LpVariable.matrix("gama", range(4), cat=pl.LpBinary)
E = pl.LpVariable.matrix("E", (range(4), range(4)), cat=pl.LpContinuous)

A_k = np.empty([4, 4, 4])
A_k[0] = np.eye(4)
A_k[1] = A
A_k[2] = A @ A_k[1]
A_k[3] = A @ A_k[2]

U = np.empty([4, 4, 1])
U[0] = np.empty([4, 1])
U[1] = A_k[0] @ B * u[0]
U[2] = A_k[1] @ B * u[1]
U[3] = A_k[2] @ B * u[2]

print(U)

U1 = np.empty([4, 4])
for i in range(4):
    for j in range(4):
     U1[i][j] = U[i][j][0]


# Add the constraints to the model
# i: time step
# j: dimension
for i in range(4):
    for j in range(4):
        model += (y[i][j] - U1[i][j] - (pl.lpSum([A_k[i][j][a] * x[a] for a in range(4)])) - E[i][j] <= delta[i][j])
        model += (y[i][j] - U1[i][j] - (pl.lpSum([A_k[i][j][a] * x[a] for a in range(4)])) - E[i][j] >= -delta[i][j])
        model += (E[j][i] <= gama[i])
        model += (E[j][i] >= -gama[i])

# Add the objective function to the model
model += pl.lpSum([gama[0], gama[1], gama[2], gama[3]])

# Solve the problem
status = model.solve()

print(f"status: {model.status}, {pl.LpStatus[model.status]}")

print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name}: {var.value()}")

print(states[t])

# print("#######")
# print(A @ states[-50])
# print(A[1][0] * states[-50][0] + A[1][1] * states[-50][1] + A[1][2] * states[-50][2] + A[1][3] * states[-50][3])
# print("#######")
# print(delta)