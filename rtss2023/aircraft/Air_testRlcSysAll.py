from Air_SysAllDim import SystemALLDim
from utils.Baseline import F16
from utils.Baseline import Boeing
from utils.Baseline import AircraftPitch
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(3) * 0.001
detector = window(tao, 3, 10)
exp = AircraftPitch
attack = [np.array([0.00])] * 30
attack_duration = 30
for i in range(attack_duration):
    attack[i] = 0.05 * i

sys = SystemALLDim(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
# print("max_index: ", max_index)
x_hat_arr = [x[2] for x in sys.y_hat]
x_tilda_arr = [x[2] for x in sys.y_tilda]
x_low = []
x_up = []
for i in range(len(x_hat_arr) - 1):
    x_low.append(x_hat_arr[i] + sys.theta[i][2][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][2][1])
tao_arr0 = [x[0] for x in sys.taos]
tao_arr1 = [x[1] for x in sys.taos]
tao_arr2 = [x[2] for x in sys.taos]

# print(sys.theta[:, 0, 0])
# print(sys.theta[:, 0, 1])
# print(sys.taos)

# t0 = [x[0] for x in sys.y_hat]
# print("max 0", np.max(t0))
# print("min 0", np.min(t0))
# t0 = [x[1] for x in sys.y_hat]
# print("max 1", np.max(t0))
# print("min 1", np.min(t0))
# t0 = [x[2] for x in sys.y_hat]
# print("max 2", np.max(t0))
# print("min 2", np.min(t0))
# t0 = [x[3] for x in sys.y_hat]
# print("max 3", np.max(t0))
# print("min 3", np.min(t0))
# t0 = [x[4] for x in sys.y_hat]
# print("max 4", np.max(t0))
# print("min 4", np.min(t0))
# print(sys.y_tilda[400])

reach = [x for x in sys.klevels]
print(sys.numAdjust)
print(sys.avgAdj * 1.5 * sys.numAdjust/sys.i)

plt.figure()
plt.subplot(4, 2, 1)
plt.plot(x_low, c='red', linestyle=':', label='x_low')
plt.plot(x_up, c='red', linestyle=':', label='x_up')
plt.plot(x_tilda_arr, c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(4, 2, 2)
plt.plot(reach[0:-1], c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(4, 2, 3)
plt.plot(tao_arr0, c='blue', linestyle=':', label='tao')

plt.subplot(4, 2, 4)
plt.plot(tao_arr1, c='blue', linestyle=':', label='tao')

plt.subplot(4, 2, 5)
plt.plot(tao_arr2, c='blue', linestyle=':', label='tao')
plt.show()
