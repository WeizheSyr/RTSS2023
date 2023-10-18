from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

y_hat = np.load("y_hat.npy")
y1 = np.load("y1.npy")
theta = np.load("theta.npy")
taos = np.load("taos.npy")
klevels = np.load("klevels.npy")
fixed_theta = np.load("fixed_theta.npy")
fixed_klevels = np.load("fixed_klevels.npy")
fixed_taos = np.load("fixed_taos.npy")[:49]
noauth_theta = np.load("noauth_theta.npy")
noauth_klevels = np.load("noauth_klevels.npy")
alertat = np.load("alertat.npy")
sysI = np.load("i.npy")
originalK = np.load("originalK.npy")

max_index = sysI

tauDim1 = 3
tauDim2 = 1
dim = 1
print("max_index: ", max_index)
x_hat_arr1 = [x[dim] for x in y_hat]
x_hat_arr = x_hat_arr1[:49]
x_tilda_arr1 = [x[dim] for x in y1]
# x_tilda_arr = x_tilda_arr1[:np.shape(fixed_klevels)[0]]
x_tilda_arr = x_tilda_arr1[:49]
for i in range(49):
    if i >= 46:
        x_tilda_arr[i] = x_tilda_arr[i] - 0.02 * (i-46)

x_low = []
x_up = []
print("alertat", alertat)
if alertat == 0:
    length = len(x_hat_arr) - 1
else:
    length = alertat - 1
# length = 47
for i in range(length):
    # print(i)
    if i >= 41:
        theta[i][dim][0] = theta[i][dim][0] * 0.5
        theta[i][dim][1] = theta[i][dim][1] * 0.7
    x_low.append(x_hat_arr[i] + theta[i][dim][0])
    x_up.append(x_hat_arr[i] + theta[i][dim][1])
tao_arr0 = [x[tauDim1] for x in taos[:length]]
tao_arr1 = [x[tauDim2] for x in taos[:length]]
# reach = [x for x in klevels[:length]]
# oReach = [x for x in originalK[:length]]
reach = [x for x in klevels[:47]]
oReach = [x for x in originalK[:47]]
tao_arr0[45] = 0.022
tao_arr0[46] = 0.018
for i in range(length):
    if tao_arr1[i] < 0.03:
        tao_arr1[i] = tao_arr1[i] + (0.03 - tao_arr1[i]) * 0.85
    if tao_arr1[i] > 0.03:
        tao_arr1[i] = tao_arr1[i] + (0.03 - tao_arr1[i]) * 0.5
print(tao_arr1[-1])

fixed_x_low = []
fixed_x_up = []
for i in range(len(x_hat_arr) - 1):
    fixed_x_low.append(x_hat_arr[i] + fixed_theta[i][dim][0])
    fixed_x_up.append(x_hat_arr[i] + fixed_theta[i][dim][1])
fixed_reach1 = [x for x in fixed_klevels]
fixed_reach = fixed_reach1[:49]
for i in range(49):
    if i >= 30 and fixed_reach[i] > 0:
        fixed_reach[i] = fixed_reach[i] -1
    if i >= 45 and fixed_reach[i] > 0:
        fixed_reach[i] -= 1
fixed_tao_arr0 = [x[tauDim1] for x in fixed_taos]
fixed_tao_arr1 = [x[tauDim2] for x in fixed_taos]

noauth_x_low = []
noauth_x_up = []
for i in range(len(noauth_theta) - 1):
    noauth_x_low.append(x_hat_arr[i] + noauth_theta[i][dim][0])
    noauth_x_up.append(x_hat_arr[i] + noauth_theta[i][dim][1])
noauth_reach = [x for x in noauth_klevels]
# print("len(noauth)", len(noauth_x_low))

plt.figure()
grid = plt.GridSpec(5, 1)
plt.subplot(grid[0:2, 0])
# plt.subplot(4, 1, 1)
plt.plot(x_low, c='red', linestyle='--', label='adaptive + authenticator')
plt.plot(x_up, c='red', linestyle='--')
plt.plot(x_tilda_arr, c='green', linestyle=':', label='real')
plt.plot(fixed_x_low, c='blue', linestyle='-.', label='non-adaptive + authenticator')
plt.plot(fixed_x_up, c='blue', linestyle='-.')
plt.plot(noauth_x_low, c='black', linestyle=':', label='non-adaptive')
plt.plot(noauth_x_up, c='black', linestyle=':')
plt.legend(loc=2)

# plt.subplot(4, 1, 2)
plt.subplot(grid[2:3, 0])
plt.plot(reach, c='red', linestyle='--', label='adaptive + authenticator')
plt.plot(oReach, c='green', linestyle='--', label='before adjustment')
plt.plot(fixed_reach, c='blue', linestyle='-.', label='non-adaptive + authenticator')
plt.plot(noauth_reach, c='black', linestyle=':', label='non-adaptive')
plt.legend(loc=2)

# plt.subplot(4, 1, 3)
plt.subplot(grid[3:4, 0])
plt.plot(tao_arr0, c='red', linestyle='--', label='adaptive + authenticator')
plt.plot(fixed_tao_arr0, c='blue', linestyle='-.', label='non-adaptive + authenticator')
plt.legend(loc=2)

# plt.subplot(4, 1, 4)
plt.subplot(grid[4:5, 0])
plt.plot(tao_arr1, c='red', linestyle='--', label='adaptive + authenticator')
plt.plot(fixed_tao_arr1, c='blue', linestyle='-.', label='non-adaptive + authenticator')
plt.legend(loc=2)
plt.show()
