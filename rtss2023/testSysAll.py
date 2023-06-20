from system import System
from system1 import System1
from systemALLDim import SystemALLDim
from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(7) * 0.02
# tao = [0.5] * 7
detector = window(tao, 7, 10)
exp = Platoon
# attack = [np.array([0.01])] * 20
attack_duration = 20
attack = np.zeros(attack_duration)
for i in range(attack_duration):
    attack[i] = 0.005 * i

sys = SystemALLDim(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = len(sys.est1)
print(max_index)
t_arr = np.linspace(0, 10, max_index)
ref = [x[0] for x in sys.reference_list[:max_index + 1]]
print(ref)
# print(real_y_arr)
y_arr = [x[0] for x in sys.y_tilda1]
y_arr1 = [x[0] for x in sys.y1]
print(y_arr)
# print(y_arr)
v1_arr = [x for x in sys.est1]
v2_arr = [x for x in sys.est2]
print(v1_arr)
print(v2_arr)
# p_arr = [x[0] for x in sys.predict_list[:max_index + 1]]

# y_arr = [x for x in sys.residual_list[:max_index + 1]]
# num = 0
# for i in y_arr:
#     if i > 0.06:
#         num = num + 1
# print(num)
# bound = []
# for i in range(len(sys.reference_list[:max_index + 1]) - len(sys.lowbound)):
#     bound.append(0)
#
# for i in range(len(sys.lowbound)):
#     bound.append(abs(sys.lowbound[i] - sys.upbound[i]))

plt.plot(y_arr1, c='orange', linestyle=':', label='y_arr')
plt.plot(y_arr, c='yellow', linestyle=':', label='y_arr')
plt.plot(v1_arr, c='blue', linestyle=':', label='y_arr')
plt.plot(v2_arr, c='red', linestyle=':', label='y_arr')

plt.show()
