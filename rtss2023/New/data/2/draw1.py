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
x1 = x[30: ]
x_arr = [x[0] for x in x1]
for i in range(10):
    if x_arr[i] > x_arr[10]:
        x_arr[i] = (x_arr[i] - x_arr[10])/2

plt.figure(figsize=(7,3))
plt.plot(x_arr, c='blue', linestyle='-')

plt.scatter(45, x_arr[45], color='orange', marker='o', s=100, label='alarm(Ours)')
plt.scatter(32, x_arr[32], color='purple', marker='^', s=100, label='alarm(fixed)')
plt.scatter(43, x_arr[43], color='green', marker='v', s=100, label='alarm(CUSUM)')
plt.axvline(x=10, color='red', linestyle='--')
plt.axhline(y=x_arr[8], color='grey', linestyle='--')

plt.legend()
plt.show()

