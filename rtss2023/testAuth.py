from Authenticate import Authenticate
from simulators.linear.platoon import Platoon
import numpy as np
import os.path

if __name__ == "__main__":
    max_index = 700
    dt = 0.02
    # ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    ref = [np.array([1])] * 301 + [np.array([1])] * 400
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(7) * -0.002,
                      'up': np.ones(7) * 0.002}
        },
        'measurement': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(7) * -0.002,
                      'up': np.ones(7) * 0.002}
        }
    }
    # lk = LaneKeeping('test', dt, max_index, noise)
    lk = Platoon('test', dt, max_index, noise)

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

    t = -70
    m = 7
    n = 4
    timestep = 14
    u = np.zeros([timestep, n])
    for i in range(timestep):
        u[i] = inputs[t + i]
    y = np.zeros([timestep, m])
    for i in range(timestep):
        y[i] = feedbacks[t + i]

    auth = Authenticate(lk, n)
    auth.getInputs(u)
    auth.getFeedbacks(y)

    auth.getAuth()
    print(auth.x)
    auth.getAllBound()
    print(auth.bound)