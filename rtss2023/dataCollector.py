from system1 import System1
import matplotlib.pyplot as plt
from simulators.linear.lane_keeping import LaneKeeping
from simulators.linear.platoon import Platoon
import numpy as np
import system as sys
from utils import Simulator

if __name__ == "__main__":
    max_index = 600
    dt = 0.02
    # ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    ref = [np.array([1])] * 301 + [np.array([1])] * 300
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(7) * 0.01}
        },
        'measurement': {
            'type': 'white',
            'param': {'C': np.eye(7) * 0.01}
        }
    }
    # lk = LaneKeeping('test', dt, max_index, noise)
    lk = Platoon('test', dt, max_index, noise)

    print('A')
    print(lk.sysd.A)
    print('B')
    print(lk.sysd.B)
    print('C')
    print(lk.C)

    start_point = 500
    delt_x = 1

    for i in range(0, max_index + 1):
        assert lk.cur_index == i
        lk.update_current_ref(ref[i])

        if i > start_point:
            # lk.cur_feedback[0] += delt_x
            lk.cur_feedback[1] *= 2
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
    y_arr = [x[0] for x in lk.states[:max_index + 1]]

    plt.plot(ref, c='orange', label='y_arr')
    plt.plot(y_arr, c='blue', label='y_arr')
    # plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()