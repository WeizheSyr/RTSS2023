from system import System
from utils.Baseline import rlc_circuit_bias
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

detector = window(10, 0.2)
exp = rlc_circuit_bias
# attack = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
attack = [0]*50
attack_duration = len(attack)

sys = System(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = len(sys.index_list)
print(max_index)
t_arr = np.linspace(0, 10, max_index)
ref = [x[0] for x in sys.reference_list[:max_index + 1]]
print(ref)
# real_y_arr = [x for x in sys.real_y[:max_index + 1]]
# print(real_y_arr)
# y_arr = [x[0] for x in sys.y_list[:max_index + 1]]
# print(y_arr)
v_arr = [x for x in sys.v1[:max_index + 1]]
print(v_arr)
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


# plt.plot(t_arr, y_arr, c='blue', linestyle=':', label='y_arr')
# plt.plot(t_arr, ref, c='red', linestyle='-', label='ref')
# plt.plot(t_arr, real_y_arr, c='green', linestyle='--', label='real_y_arr')
# plt.plot(t_arr, p_arr, c='yellow', linestyle=':', label='bound')
plt.plot(v_arr, c='blue', linestyle=':', label='y_arr')

plt.show()