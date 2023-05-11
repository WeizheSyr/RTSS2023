from system1 import System1
import matplotlib.pyplot as plt
from simulators.linear.lane_keeping import LaneKeeping
import numpy as np
import system as sys
from utils import Simulator

if __name__ == "__main__":
    max_index = 600
    dt = 0.01
    ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.001}
        }
    }
    lk = LaneKeeping('test', dt, max_index, noise)

    print('A')
    print(lk.sysc.A)
    print('B')
    print(lk.sysc.B)
    print('C')
    print(lk.C)

    start_point = 550
    delt_x = 0.001

    for i in range(0, max_index + 1):
        assert lk.cur_index == i
        lk.update_current_ref(ref[i])

        if i > start_point:
            lk.cur_feedback[0] += delt_x
            # print(lk.cur_feedback[0])

        lk.evolve()
        # print("###################")
        # print(lk.cur_x[0])

    import matplotlib.pyplot as plt

    print(lk.inputs[-1].shape)
    print(lk.outputs[0])
    print(lk.feedbacks.size)
    # np.savetxt(f'save/inputs_{lk.name}.csv', lk.inputs[500:], delimiter=',')
    # np.savetxt(f'save/states_{lk.name}.csv', lk.states[500:], delimiter=',')
    # np.savetxt(f'save/feedbacks_{lk.name}.csv', lk.feedbacks[500:], delimiter=',')

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in lk.refs[:max_index + 1]]
    y_arr = [x[0] for x in lk.feedbacks[:max_index + 1]]

    plt.plot(ref, c='orange', label='y_arr')
    plt.plot(y_arr, c='yellow', label='y_arr')
    # plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()