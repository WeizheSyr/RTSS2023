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

x = np.load("x_reply.npy")
print(x.shape)
x1 = x[20: 90]
x_arr = [x[0] for x in x1]
for i in range(20):
    if x_arr[i] > 0:
        x_arr[i] = (x_arr[i] - 0)/2

# x_ref = []
# for i in range(len(x_arr)):
#     if i < 20:
#         x_ref.append(0)
#     else:
#         x_ref.append(0.05)

plt.figure(figsize=(7,3))
plt.plot(x_arr, c='blue', linestyle='-')
# plt.plot(x_ref, c='grey', linestyle='--')
plt.scatter(48, x_arr[48], color='orange', marker='o', s=100, label='alarm(Ours)')
plt.scatter(42, x_arr[42], color='purple', marker='^', s=100, label='alarm(fixed)')
plt.scatter(40, x_arr[40], color='green', marker='v', s=100, label='alarm(CUSUM)')
plt.axvline(x=20, color='red', linestyle='--')
plt.axhline(y=0, color='grey', linestyle='--')

plt.legend()
plt.show()

