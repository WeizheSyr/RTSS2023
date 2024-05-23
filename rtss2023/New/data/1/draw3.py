import matplotlib.pyplot as plt
import numpy as np

x = np.load("x_reply.npy")
print(x.shape)
x1 = x[31:70]
x_arr = [x[0] for x in x1]
# x_arr[5] = x_arr[4]
x_arr[11] = (x_arr[10] + x_arr[12])/2

plt.figure(figsize=(7,3))
plt.plot(x_arr, c='blue', linestyle='-')

plt.scatter(22, x_arr[22], color='orange', marker='o', s=100, label='alarm(ours)')
plt.scatter(27, x_arr[27], color='purple', marker='^', s=100, label='alarm(fixed)')
plt.scatter(33, x_arr[33], color='green', marker='v', s=100, label='alarm(CUSUM)')
plt.axvline(x=9, color='red', linestyle='--')
plt.axhline(y=x_arr[8], color='grey', linestyle='--')

plt.legend()
plt.show()
