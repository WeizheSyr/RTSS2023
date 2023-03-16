# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7322210&tag=1
import numpy as np
from numpy.linalg import inv

"""
for any column vectors, the covariance matrix
    Cov(x) = E[ZZT] - (E[Z])(E[Z])T
"""

class chi_square():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.Z_k = None
        self.name = 'chi_square'
        self.g = 0
        self.inv = 0
    # def columnCon(self, z):
    #     cov =

    def detect(self, z_k):
        z_k = z_k[:, np.newaxis]
        if self.Z_k is None:
            self.Z_k = z_k
            # self.Z_k = self.Z_k[:, np.newaxis]
        else:
            self.Z_k = np.append(self.Z_k, z_k,  axis=1)
        if self.Z_k[0].size == 1:
            return False


        # z_k = np.reshape(z_k, (z_k.size, 1))
        # print(z_t.shape)
        P_k = np.cov(self.Z_k, rowvar=True, bias=True)
        # P_k = np.reshape(P_k, (-1, 1))
        inv_ma = inv(P_k)
        # print(z_t.T.shape)
        self.g = z_k.T @ inv_ma @ z_k
        # print(g_k)
        # self.g = g_k
        self.P_k = inv(P_k)
        self.inv = inv_ma


        if self.g > self.threshold:
            # print(self.g)
            return True
        else:
            return False

if __name__ == '__main__':
    x = np.ones([3, 300]) * 10
    detector = chi_square(threshold=16)
    x_hat = np.ones([3, 300]) *10 + np.random.rand(3, 300) * 0.000001
    # print(x_hat)
    for i in range(0, x[0].size):
        z_t = x_hat[:, i] - x[:, i]
        # print(z_t)
        # z_t.reshape(3, 1)
        alarm = detector.detect((z_t))
        if alarm:
            print(f"raise alarm at {i}, z_t:{z_t}, residual:{detector.g}")
            # break

