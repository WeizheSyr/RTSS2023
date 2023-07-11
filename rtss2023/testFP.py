from systemALLDim import SystemALLDim
from EvaluationFixedDetector import SystemFixedDetector
from ThreeDetector import ThreeDetector
from utils.Baseline import Platoon
from utils.detector.windowBased import window
from FPEvaluation import FPEvaluation
import matplotlib.pyplot as plt
import numpy as np
import time

# tao = np.ones(7) * 0.1
# tao = [0.5] * 7
# detector = window(tao, 7, 10)
# fixed_tao = np.ones(7) * 0.1
# fixed_detector = window(fixed_tao, 7, 10)
# noauth_detector = window(fixed_tao, 7, 10)
# exp = Platoon
# attack = np.zeros(50)
# attack_duration = 50
# attack = np.zeros(attack_duration)

# sys = ThreeDetector(detector=detector, fixed_detector=fixed_detector, noauth_detector=noauth_detector, exp=exp, attack=attack, attack_duration=attack_duration)

FP_our = 0
FP_fixed = 0
largerThanK = 0
totalLength = 0

for i in range(100):
    # rseed = np.uint32(int(time.time()))
    # print(rseed)
    # np.random.seed(rseed)
    print("new")
    tao = np.ones(7) * 0.1
    detector = window(tao, 7, 10)
    fixed_tao = np.ones(7) * 0.1
    fixed_detector = window(fixed_tao, 7, 10)
    exp = Platoon
    attack = np.zeros(50)
    attack_duration = 50

    sys = FPEvaluation(detector=detector, fixed_detector=fixed_detector, exp=exp, attack=attack, attack_duration=attack_duration)
    print(sys.i)
    totalLength = totalLength + len(sys.fixed_klevels)
    for j in range(len(sys.fixed_klevels)):
        if sys.fixed_klevels[j] >= sys.klevel - 1:
            largerThanK = largerThanK + 1

    print("totalLength", totalLength)
    print("k")
    if len(sys.klevels) < 100:
        FP_our = FP_our + 1
    if len(sys.fixed_klevels) < 100:
        FP_fixed = FP_fixed + 1

    del sys
    print("**********")
    del attack_duration
    del attack
    del exp
    del fixed_detector
    del fixed_tao
    del detector
    del tao
print("FP_our", FP_our)
print("FP_fixed", FP_fixed)
print("largerK", largerThanK / totalLength)

# max_index = sys.i
# dim = 0
# print("max_index: ", max_index)
# x_hat_arr = [x[dim] for x in sys.y_hat]
# x_tilda_arr = [x[dim] for x in sys.y1]
# y_tilda
# y12
# x_low = []
# x_up = []
# if sys.alertat == 0:
#     length = len(x_hat_arr) - 1
# else:
#     length = sys.alertat - 1
# for i in range(length):
#     x_low.append(x_hat_arr[i] + sys.theta[i][dim][0])
#     x_up.append(x_hat_arr[i] + sys.theta[i][dim][1])
# tao_arr0 = [x[0] for x in sys.taos[:length]]
# tao_arr1 = [x[4] for x in sys.taos[:length]]
# reach = [x for x in sys.klevels[:length]]
#
# fixed_x_low = []
# fixed_x_up = []
# for i in range(len(x_hat_arr) - 1):
#     fixed_x_low.append(x_hat_arr[i] + sys.fixed_theta[i][dim][0])
#     fixed_x_up.append(x_hat_arr[i] + sys.fixed_theta[i][dim][1])
# fixed_reach = [x for x in sys.fixed_klevels]
# fixed_tao_arr0 = [x[0] for x in sys.fixed_taos]
# fixed_tao_arr1 = [x[4] for x in sys.fixed_taos]
#
# noauth_x_low = []
# noauth_x_up = []
# for i in range(len(sys.noauth_theta) - 1):
#     noauth_x_low.append(x_hat_arr[i] + sys.noauth_theta[i][dim][0])
#     noauth_x_up.append(x_hat_arr[i] + sys.noauth_theta[i][dim][1])
# noauth_reach = [x for x in sys.noauth_klevels]
# print("len(noauth)", len(noauth_x_low))

# plt.figure()
# grid = plt.GridSpec(5, 1)
# plt.subplot(grid[0:2, 0])
# # plt.subplot(4, 1, 1)
# plt.plot(x_low, c='red', linestyle='--', label='adaptive + authenticator')
# plt.plot(x_up, c='red', linestyle='--')
# plt.plot(x_tilda_arr, c='green', linestyle=':', label='real')
# plt.plot(fixed_x_low, c='blue', linestyle='-.', label='non-adaptive + authenticator')
# plt.plot(fixed_x_up, c='blue', linestyle='-.')
# plt.plot(noauth_x_low, c='black', linestyle=':', label='non-adaptive')
# plt.plot(noauth_x_up, c='black', linestyle=':')
# plt.legend(loc=2)
#
# # plt.subplot(4, 1, 2)
# plt.subplot(grid[2:3, 0])
# plt.plot(reach, c='red', linestyle='--', label='adaptive + authenticator')
# plt.plot(fixed_reach, c='blue', linestyle='-.', label='non-adaptive + authenticator')
# plt.plot(noauth_reach, c='black', linestyle=':', label='non-adaptive')
# plt.legend(loc=2)
#
# # plt.subplot(4, 1, 3)
# plt.subplot(grid[3:4, 0])
# plt.plot(tao_arr0, c='red', linestyle='--', label='adaptive + authenticator')
# plt.plot(fixed_tao_arr0, c='blue', linestyle='-.', label='non-adaptive + authenticator')
# plt.legend(loc=2)
#
# # plt.subplot(4, 1, 4)
# plt.subplot(grid[4:5, 0])
# plt.plot(tao_arr1, c='red', linestyle='--', label='adaptive + authenticator')
# plt.plot(fixed_tao_arr1, c='blue', linestyle='-.', label='non-adaptive + authenticator')
# plt.legend(loc=2)
# plt.show()
