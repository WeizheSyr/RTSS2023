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
print(inputs[-2].shape)
print(states[-1])
print(feedbacks[-2])