from Lane_sysAllDim import SystemALLDim
from utils.Baseline import Lane
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

tao = np.ones(4) * 0.02
# tao = [0.5] * 7
detector = window(tao, 4, 10)
exp = Lane
attack = [np.array([0.00])] * 30
attack_duration = 30
# attack = np.zeros(attack_duration)
# for i in range(attack_duration):
#     attack[i] = 0.005 * i

sys = SystemALLDim(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
# print("max_index: ", max_index)
x_hat_arr = [x[0] for x in sys.y_hat]
x_tilda_arr = [x[0] for x in sys.y_tilda]
x_low = []
x_up = []
for i in range(len(x_hat_arr) - 1):
    x_low.append(x_hat_arr[i] + sys.theta[i][0][0])
    x_up.append(x_hat_arr[i] + sys.theta[i][0][1])
tao_arr0 = [x[0] for x in sys.taos]
tao_arr1 = [x[1] for x in sys.taos]
tao_arr2 = [x[2] for x in sys.taos]
tao_arr3 = [x[3] for x in sys.taos]

# print(sys.theta[:, 0, 0])
# print(sys.theta[:, 0, 1])
# print(sys.taos)

reach = [x for x in sys.klevels]

avg_adj = sum(sys.time_adjs) / len(sys.time_adjs)
avg_auth = sum(sys.time_auths) / len(sys.time_auths)
avg_reco = sum(sys.time_recovers) / len(sys.time_recovers)
avg_err = sum(sys.time_errors) / len(sys.time_errors)
print("avg_adj", avg_adj)
print("avg_auth", avg_auth)
print("avg_reco", avg_reco)
print("avg_err", avg_err)

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

plt.subplot(4, 2, 6)
plt.plot(tao_arr3, c='blue', linestyle=':', label='tao')
plt.show()
