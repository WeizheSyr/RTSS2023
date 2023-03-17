from system import System
from utils.Baseline import rlc_circuit_bias
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

detector = window(10, 1)
exp = rlc_circuit_bias
attack = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
attack_duration = len(attack)

sys = System(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = len(sys.index_list)
t_arr = np.linspace(0, 10, max_index + 1)
ref = [x[0] for x in sys.reference_list[:max_index + 1]]
real_y_arr = [x[0] for x in sys.real_y[:max_index + 1]]
y_arr = [x[0] for x in sys.y_list[:max_index + 1]]

plt.plot(t_arr, y_arr, t_arr, ref, t_arr, real_y_arr)
plt.show()
