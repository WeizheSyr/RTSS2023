import matplotlib.pyplot as plt
import numpy as np

x = np.load("x_delay.npy")
print(x.shape)
x1 = x[31:70]
x_arr = [x[0] for x in x1]
x_arr[5] = x_arr[4]
x_arr[11] = (x_arr[10] + x_arr[12])/2

x_ref = []
for i in range(len(x_arr)):
    if i < 9:
        x_ref.append(x_arr[8])
    else:
        x_ref.append(0.9870)

plt.figure(figsize=(7,3))
plt.plot(x_arr, c='blue', linestyle='-')
plt.plot(x_ref, c='grey', linestyle='--')

plt.scatter(24, x_arr[24], color='orange', marker='o', s=100, label='alarm(ours)')
# plt.scatter(color='purple', marker='^', s=100, label='alarm(fixed)')
plt.scatter(37, x_arr[37], color='green', marker='v', s=100, label='alarm(CUSUM)')
plt.axvline(x=9, color='red', linestyle='--')
# plt.axhline(y=x_arr[8], color='grey', linestyle='--')

y_min, y_max = min(x_arr), max(x_arr)
yticks = np.linspace(y_min, y_max, 5)
yticks = [0.984, 0.985, 0.986, 0.987, 0.988]
plt.yticks(yticks)

plt.legend()
plt.show()
