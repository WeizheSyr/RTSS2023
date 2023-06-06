from system1 import System1
import matplotlib.pyplot as plt
from simulators.linear.quadrotor import Quadrotor
import numpy as np
import system as sys
from utils import Simulator

if __name__ == "__main__":
    max_index = 700
    dt = 0.02
    # ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    ref = [np.array([2])] * 301 + [np.array([2])] * 400
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
    # lk = LaneKeeping('test', dt, max_index, noise)
    lk = Quadrotor('test', dt, max_index, noise)

    print('A')
    print(lk.sysd.A)
    print('B')
    print(lk.sysd.B)
    print('C')
    print(lk.C)

    start_point = 600
    delt_x = 0.2

    for i in range(0, max_index + 1):
        assert lk.cur_index == i
        lk.update_current_ref(ref[i])

        if i > start_point:
            lk.cur_feedback[5] += delt_x
            # lk.cur_feedback[1] *= 2
            # print(lk.cur_feedback[0])

        lk.evolve()
        # print("###################")
        # print(lk.cur_x[0])

    import matplotlib.pyplot as plt

    print(lk.inputs[-1].shape)
    print(lk.outputs[-1])
    print(lk.feedbacks.size)
    np.savetxt(f'save/inputs_{lk.name}.csv', lk.inputs[400:], delimiter=',')
    np.savetxt(f'save/states_{lk.name}.csv', lk.states[400:], delimiter=',')
    np.savetxt(f'save/feedbacks_{lk.name}.csv', lk.feedbacks[400:], delimiter=',')

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in lk.refs[:max_index + 1]]
    y_arr = [x[5] for x in lk.outputs[:max_index + 1]]

    plt.plot(ref, c='orange', label='y_arr')
    plt.plot(y_arr, c='blue', label='y_arr')
    # plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()