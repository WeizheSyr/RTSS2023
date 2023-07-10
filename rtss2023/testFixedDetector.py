from systemALLDim import SystemALLDim
from EvaluationFixedDetector import SystemFixedDetector
from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(7) * 0.04
# tao = [0.5] * 7
detector = window(tao, 7, 10)
fixed_tao = np.ones(7) * 0.07
fixed_detector = window(fixed_tao, 7, 10)
exp = Platoon
attack = [np.array([0.005])] * 10 + [np.array([0.003])] * 10 + [np.array([0.005])] * 10
attack_duration = 30
# attack = np.zeros(attack_duration)
for i in range(attack_duration):
    attack[i] = 0.003 * i

sys = SystemFixedDetector(detector=detector, fixed_detector=fixed_detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
print("max_index: ", max_index)
x_hat_arr = [x[0] for x in sys.y_hat]
x_tilda_arr = [x[0] for x in sys.y_tilda]
x_low = []
x_up = []
for i in range(len(x_hat_arr) - 1):
    x_low.append(x_hat_arr[i] + sys.theta[i][0][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][0][1])
tao_arr0 = [x[0] for x in sys.taos]
tao_arr1 = [x[4] for x in sys.taos]
reach = [x for x in sys.klevels]

fixed_x_low = []
fixed_x_up = []
for i in range(len(x_hat_arr) - 1):
    fixed_x_low.append(x_hat_arr[i] + sys.fixed_theta[i][0][0])
    fixed_x_up.append(x_hat_arr[i] + sys.fixed_theta[i][0][1])
fixed_reach = [x for x in sys.fixed_klevels]
fixed_tao_arr0 = [x[0] for x in sys.fixed_taos]
fixed_tao_arr1 = [x[4] for x in sys.fixed_taos]


plt.figure()
plt.subplot(4, 1, 1)
plt.plot(x_low, c='red', linestyle=':', label='x_low')
plt.plot(x_up, c='red', linestyle=':', label='x_up')
plt.plot(x_tilda_arr, c='green', linestyle=':', label='x_tilda_arr')
plt.plot(fixed_x_low, c='blue', linestyle=':', label='fixed_x_low')
plt.plot(fixed_x_up, c='blue', linestyle=':', label='fixed_x_up')

plt.subplot(4, 1, 2)
plt.plot(reach, c='red', linestyle=':', label='klevels')
plt.plot(fixed_reach, c='blue', linestyle=':', label='fixed_klevels')

plt.subplot(4, 1, 3)
plt.plot(tao_arr0, c='red', linestyle=':', label='tao')
plt.plot(fixed_tao_arr0, c='blue', linestyle=':', label='fixed_tao')

plt.subplot(4, 1, 4)
plt.plot(tao_arr1, c='red', linestyle=':', label='tao')
plt.plot(fixed_tao_arr1, c='blue', linestyle=':', label='fixed_tao')
plt.show()
