from timeConsuming import TimeConsuming
from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np
import time

tao = np.ones(7) * 0.05
# tao = [0.5] * 7
detector = window(tao, 7, 10)
exp = Platoon
attack = [np.array([0.00])] * 30
attack_duration = 30

sys = TimeConsuming(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

# time
sumTimeAll = sys.timeAll
print("sumTimeAll", sumTimeAll)

sumTimeAuth = 0
for i in range(len(sys.timeAuth)):
    sumTimeAuth = sumTimeAuth + sys.timeAuth[i]
print("sumTimeAuth", sumTimeAuth)
print("each Auth", sumTimeAuth / len(sys.timeAuth))

sumTimeQuick = 0
for i in range(len(sys.timeQuick)):
    sumTimeQuick = sumTimeQuick + sys.timeQuick[i]
print("sumTimeQuick", sumTimeQuick)
print("each Quick", sumTimeQuick / len(sys.timeQuick))

sumTimeReach = 0
for i in range(len(sys.timeReach)):
    sumTimeReach = sumTimeReach + sys.timeReach[i]
print("sumTimeReach", sumTimeReach)
print("each Reach", sumTimeReach / len(sys.timeReach))

sumTimeAdjust = 0
for i in range(len(sys.timeAdjust)):
    sumTimeAdjust = sumTimeAdjust + sys.timeAdjust[i]
print("sumTimeAdjust", sumTimeAdjust)
print("each Adjust", sumTimeAdjust / len(sys.timeAdjust))

sumTimeDetect = 0
for i in range(len(sys.timeDetect)):
    sumTimeDetect = sumTimeDetect + sys.timeDetect[i]
print("sumTimeDetect", sumTimeDetect)
print("each detect", sumTimeDetect / len(sys.timeDetect))

max_index = sys.i

print("Quick in one time step", sumTimeQuick / max_index)
print("Reach in one time step", sumTimeReach / max_index)
print("Adjust in one time step", sumTimeAdjust / max_index)
print("Detect in one time step", sumTimeDetect / max_index)
print("max_index", max_index)

# print("max_index: ", max_index)
x_hat_arr = [x[0] for x in sys.y_hat]
x_tilda_arr = [x[0] for x in sys.y_tilda]
x_low = []
x_up = []
for i in range(len(x_hat_arr) - 1):
    x_low.append(x_hat_arr[i] + sys.theta[i][0][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][0][1])
tao_arr0 = [x[1] for x in sys.taos]
tao_arr1 = [x[4] for x in sys.taos]

# print(sys.theta[:, 0, 0])
# print(sys.theta[:, 0, 1])
# print(sys.taos)

reach = [x for x in sys.klevels]

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x_low, c='red', linestyle=':', label='x_low')
plt.plot(x_up, c='red', linestyle=':', label='x_up')
plt.plot(x_tilda_arr, c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(2, 2, 2)
plt.plot(reach, c='blue', linestyle=':', label='x_tilda_arr')

plt.subplot(2, 2, 3)
plt.plot(tao_arr0, c='blue', linestyle=':', label='tao')

plt.subplot(2, 2, 4)
plt.plot(tao_arr1, c='blue', linestyle=':', label='tao')
plt.show()



