from Air_rlcSysAllDim import SystemALLDim
from utils.Baseline import F16
from utils.Baseline import Boeing
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(4) * 0.5
detector = window(tao, 5, 10)
exp = Boeing
attack = [np.array([0.00])] * 30
attack_duration = 30
for i in range(attack_duration):
    attack[i] = 0.05 * i

sys = SystemALLDim(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
# print("max_index: ", max_index)
x_hat_arr = [x[0] for x in sys.y_hat]
x_tilda_arr = [x[0] for x in sys.y_tilda]
# print(sys.y_tilda[150])
x_low = []
x_up = []
for i in range(len(x_hat_arr) - 1):
    x_low.append(x_hat_arr[i] + sys.theta[i][0][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][0][1])
tao_arr0 = [x[0] for x in sys.taos]
tao_arr1 = [x[0] for x in sys.taos]

# print(sys.theta[:, 0, 0])
# print(sys.theta[:, 0, 1])
# print(sys.taos)

reach = [x for x in sys.klevels]

plt.figure()
grid = plt.GridSpec(5, 1)
plt.subplot(grid[0:2, 0])
plt.plot(x_low, c='red', linestyle=':', label='x_low')
plt.plot(x_up, c='red', linestyle=':', label='x_up')
plt.plot(x_tilda_arr, c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(grid[2:3, 0])
plt.plot(reach, c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(grid[3:4, 0])
plt.plot(tao_arr0, c='blue', linestyle=':', label='tao')

# plt.subplot(grid[4:5, 0])
# plt.plot(tao_arr1, c='blue', linestyle=':', label='tao')
plt.show()
