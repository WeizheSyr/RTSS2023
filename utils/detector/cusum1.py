import numpy as np
from copy import deepcopy

class cusum:
    def __init__(self, tao, m=7, w=14, noise=0.001):
        self.m = m              # dimension of states
        self.w = w              # observation window
        self.tao = tao          # threshold tao
        self.iniTao = deepcopy(tao)
        self.queue = [[] for i in range(m)]
        self.results = np.zeros(m)
        # self.results = [0] * m  # detection results in this timestep
        self.rsum = np.zeros(m)
        self.residuals = None
        self.noise = noise
        # self.rsum = [0] * m     # sum of residual for each dimension

    # residual: residual data for this dimension
    # dim: which dimension this function detect
    def detectOneDim(self, residual, dim):
        # if len(self.queue[dim]) == self.w:
        #     self.queue[dim].pop()
        # self.queue[dim].insert(0, abs(residual) - self.noise)

        t = abs(residual) - self.noise
        if t > 0:
            self.rsum[dim] = self.rsum[dim] + t
        # print("dim, ", dim)
        # print("residualsum", sum(self.queue[dim]))

        if self.rsum[dim] > self.tao[dim]:
            return True    # alert
        else:
            return False    # no alert

    # def detectagain(self, dim):
    #     if sum(self.queue[dim]) > self.tao[dim]:
    #         return True    # alert
    #     else:
    #         return False    # no alert

    # residuals: residual data for all dimension in this timestep
    def detect(self, residuals):
        self.residuals = residuals
        for i in range(self.m):
            self.results[i] = self.detectOneDim(residuals[i], i)
        return self.results

    # def detectagain1(self, residuals):
    #     for i in range(self.m):
    #         self.results[i] = self.detectagain(i)
    #     return self.results

    def alarmOrN(self):
        if sum(self.results) >= 1:
            return True
        else:
            return False

    # def adjust(self, delta_theta):
    #     for i in range(self.m):
    #         self.tao[i] = self.tao[i] + self.tao[i] * delta_theta[i]
    #     # print("new tao", self.tao)

    def adjust(self, delta_tau, inOrDe):
        if inOrDe == 0:
            for i in range(self.m):
                self.tao[i] = self.tao[i] - delta_tau[i]
        else:
            for i in range(self.m):
                self.tao[i] = self.tao[i] + delta_tau[i]

    def minTau(self):
        t = np.array(self.tao)
        return np.argmin(t)

    def continueWork(self):
        for i in range(self.m):
            if self.detectOneDim(self.residuals[i], i):
                self.tao[i] = sum(self.queue[i])




