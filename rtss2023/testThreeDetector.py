from systemALLDim import SystemALLDim
from EvaluationFixedDetector import SystemFixedDetector
from ThreeDetector import ThreeDetector
from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(7) * 0.1
# tao = [0.5] * 7
detector = window(tao, 7, 10)
fixed_tao = np.ones(7) * 0.1
fixed_detector = window(fixed_tao, 7, 10)
noauth_detector = window(fixed_tao, 7, 10)
exp = Platoon
attack = np.zeros(50)
attack_duration = 50
# attack = np.zeros(attack_duration)
for i in range(10):
    attack[i] = 0.01 + 0.0004 * i
for i in range(10):
    attack[i + 9] = 0.014 + 0.0004 * i
for i in range(10):
    attack[i + 19] = 0.018 + 0.0004 * i
for i in range(10):
    attack[i + 29] = 0.022 + 0.0004 * i
for i in range(10):
    attack[i + 39] = 0.03 + 0.0005 * i
print("attack", attack)

sys = ThreeDetector(detector=detector, fixed_detector=fixed_detector, noauth_detector=noauth_detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
dim = 0
print("max_index: ", max_index)
x_hat_arr = [x[dim] for x in sys.y_hat]
x_tilda_arr = [x[dim] for x in sys.y1]
# y_tilda
# y12
x_low = []
x_up = []
if sys.alertat == 0:
    length = len(x_hat_arr) - 1
else:
    length = sys.alertat - 1
for i in range(length):
    x_low.append(x_hat_arr[i] + sys.theta[i][dim][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][dim][1])
tao_arr0 = [x[0] for x in sys.taos[:length]]
tao_arr1 = [x[4] for x in sys.taos[:length]]
reach = [x for x in sys.klevels[:length]]

fixed_x_low = []
fixed_x_up = []
for i in range(len(x_hat_arr) - 1):
    fixed_x_low.append(x_hat_arr[i] + sys.fixed_theta[i][dim][0])
    fixed_x_up.append(x_hat_arr[i] + sys.fixed_theta[i][dim][1])
fixed_reach = [x for x in sys.fixed_klevels]
fixed_tao_arr0 = [x[0] for x in sys.fixed_taos]
fixed_tao_arr1 = [x[4] for x in sys.fixed_taos]

noauth_x_low = []
noauth_x_up = []
for i in range(len(sys.noauth_theta) - 1):
    noauth_x_low.append(x_hat_arr[i] + sys.noauth_theta[i][dim][0])
    noauth_x_up.append(x_hat_arr[i] + sys.noauth_theta[i][dim][1])
noauth_reach = [x for x in sys.noauth_klevels]
# print("len(noauth)", len(noauth_x_low))

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(x_low, c='red', linestyle='--', label='our x')
plt.plot(x_up, c='red', linestyle='--')
plt.plot(x_tilda_arr, c='green', linestyle=':', label='real x')
plt.plot(fixed_x_low, c='blue', linestyle='-.', label='fixed detector x')
plt.plot(fixed_x_up, c='blue', linestyle='-.')
plt.plot(noauth_x_low, c='black', linestyle=':', label='no auth detector x')
plt.plot(noauth_x_up, c='black', linestyle=':')
plt.legend(loc=2)

plt.subplot(4, 1, 2)
plt.plot(reach, c='red', linestyle='--', label='our recoverability')
plt.plot(fixed_reach, c='blue', linestyle='-.', label='fixed detector recoverability')
plt.plot(noauth_reach, c='black', linestyle=':', label='no auth detector recoverability')
plt.legend(loc=2)

plt.subplot(4, 1, 3)
plt.plot(tao_arr0, c='red', linestyle='--', label='tao')
plt.plot(fixed_tao_arr0, c='blue', linestyle='-.', label='fixed detector tao')
plt.legend(loc=2)

plt.subplot(4, 1, 4)
plt.plot(tao_arr1, c='red', linestyle='--', label='tao')
plt.plot(fixed_tao_arr1, c='blue', linestyle='-.', label='fixed detector tao')
plt.legend(loc=2)
plt.show()
