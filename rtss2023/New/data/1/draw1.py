#
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.load("x.npy")
# print(x.shape)
# x1 = x[30: 55]
# x_arr = [x[0] for x in x1]
# plt.figure()
# plt.plot(x_arr, c='blue', linestyle=':', label='x')
# plt.plot(())
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

x = np.load("x.npy")
print(x.shape)
x1 = x[30: 55]
x_arr = [x[0] for x in x1]

plt.figure(figsize=(9,3))
plt.plot(x_arr, c='blue', linestyle='-')

plt.scatter(10, x_arr[10], color='orange', marker='o', s=100, label='alarm(ours)')
plt.scatter(18, x_arr[18], color='purple', marker='^', s=100, label='alarm(fixed)')
plt.scatter(22, x_arr[22], color='green', marker='v', s=100, label='alarm(CUSUM)')
plt.axvline(x=9, color='red', linestyle='--')
plt.axhline(y=x_arr[8], color='grey', linestyle='--')

plt.legend()
plt.show()

