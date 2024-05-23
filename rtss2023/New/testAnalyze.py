from newSysAnalyze import Sys
from utils.Baseline import Platoon
from utils.detector.windowBased import window
from utils.detector.cusum1 import cusum
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(7) * 0.02
detector = window(tao, 7, 10)
tao1 = np.ones(7) * 0.022
detector1 = window(tao1, 7, 10)
cusum = cusum(tao1, 7, 10, noise=0.003)
exp = Platoon
attack = [np.array([0.01])] * 30
attack_duration = 400
# attack = np.zeros(attack_duration
# for i in range(attack_duration):
#     attack[i] = 0.005 * i

sys = Sys(detector=detector, detector1=detector1, cusum=cusum ,exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
x_arr = [x[0] for x in sys.x]
# x_arr = [x[0] for x in sys.taus]
# np.save("data/1/x", sys.x)

print("FP", sys.FP/160)
# print("FP1", sys.FP1/399)
# print("FP2", sys.FP2/399)

# print("recoverTime", sys.reach.recoverTime / sys.numOfRecover)
# print("adjustTime", sys.adjustTime / sys.numOfAdjust)
# print("authTime", sys.authTime / sys.numOfAuth)
# print("errorTime", sys.errorTime / sys.numOfError)

plt.figure()
plt.plot(x_arr, c='blue', linestyle=':', label='x')
# plt.plot((sys.x[sys.alarm1st][0], sys.alarm1st), 'o', color='black')
# plt.plot((sys.x[sys.alarm1st1][0], sys.alarm1st1), 'v', color='yellow')
plt.show()
