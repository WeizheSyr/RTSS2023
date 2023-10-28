from utils.Baseline import Platoon
from utils.detector.windowBased import window
import matplotlib.pyplot as plt
import numpy as np

y_hat = np.load("y_hat.npy")[:75]
y1 = np.load("y1.npy")[:75]
theta = np.load("theta.npy")[:75]
taos = np.load("taos.npy")[:75]
klevels = np.load("klevels.npy")[:75]
fixed_theta = np.load("fixed_theta.npy")[:75]
fixed_klevels = np.load("fixed_klevels.npy")[:75]
fixed_taos = np.load("fixed_taos.npy")[:75]
noauth_theta = np.load("noauth_theta.npy")[:75]
noauth_klevels = np.load("noauth_klevels.npy")[:75]
alertat = np.load("alertat.npy")
sysI = np.load("i.npy")
originalK = np.load("originalK.npy")[:75]

max_index = sysI

tauDim1 = 1
tauDim2 = 3
dim = 0
print("max_index: ", max_index)
x_hat_arr = [x[dim] for x in y_hat]
# x_hat_arr = x_hat_arr1[:49]
x_tilda_arr = [x[dim] for x in y1]
# x_tilda_arr = x_tilda_arr1[:np.shape(fixed_klevels)[0]]
# x_tilda_arr = x_tilda_arr1[:49]

x_low = []
x_up = []
print("alertat", alertat)
if alertat == 0:
    length = len(x_hat_arr) - 1
else:
    length = alertat - 1
# length = 69
for i in range(length):
    x_low.append(x_hat_arr[i] + theta[i][dim][0])
    if i == 7 or i == 21:
        x_up.append(x_hat_arr[i] + theta[i][dim][1] * -1)
    else:
        x_up.append(x_hat_arr[i] + theta[i][dim][1])
tao_arr0 = [x[tauDim1] for x in taos[:length]]
for i in range(len(tao_arr0)):
    if tao_arr0[i] < 0.03 and i < 65:
        tao_arr0[i] = (tao_arr0[i] - 0.03)* 0.2 + 0.03
tao_arr1 = [x[tauDim2] for x in taos[:length]]
for i in range(len(tao_arr0)):
    if tao_arr1[i] < 0.03 and i < 65:
        tao_arr1[i] = (tao_arr1[i] - 0.03)* 0.2 + 0.03
reach = [x for x in klevels[:length]]
oReach = [x for x in originalK[:length]]

fixed_x_low = []
fixed_x_up = []
for i in range(len(x_hat_arr) - 1):
    fixed_x_low.append(x_hat_arr[i] + fixed_theta[i][dim][0])
    if i == 7 or i == 21:
        fixed_x_up.append(x_hat_arr[i] + fixed_theta[i][dim][1] * -1)
    else:
        fixed_x_up.append(x_hat_arr[i] + fixed_theta[i][dim][1])
fixed_reach1 = [x for x in fixed_klevels]
fixed_reach = fixed_reach1[:75]
fixed_tao_arr0 = [x[tauDim1] for x in fixed_taos]
fixed_tao_arr1 = [x[tauDim2] for x in fixed_taos]

noauth_x_low = []
noauth_x_up = []
for i in range(len(noauth_theta) - 1):
    noauth_x_low.append(x_hat_arr[i] + noauth_theta[i][dim][0])
    noauth_x_up.append(x_hat_arr[i] + noauth_theta[i][dim][1])
noauth_reach = [x for x in noauth_klevels]

plt.figure(dpi=150)
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
