import os.path
import numpy as np

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
# u_0, u_1, u_2, u_3
print("u_0, u_1, u_2, u_3")
print(inputs[-50])
print(inputs[-49])
print(inputs[-48])
print(inputs[-47])

# x_0
print("x_0")
print(states[-50])

print("y_0, y_1, y_2, y_3")
print(feedbacks[-50])
print(feedbacks[-49])
print(feedbacks[-48])
print(feedbacks[-47])

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