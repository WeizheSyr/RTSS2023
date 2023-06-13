import numpy as np


class window:
    def __init__(self, tao, m=7, w=14):
        self.m = m              # dimension of states
        self.w = w              # observation window
        self.tao = tao          # threshold tao
        self.queue = [[] for i in range(m)]
        self.results = np.zeros(m)
        # self.results = [0] * m  # detection results in this timestep
        self.rsum = np.zeros(m)
        # self.rsum = [0] * m     # sum of residual for each dimension

    # residual: residual data for this dimension
    # dim: which dimension this function detect
    def detectOneDim(self, residual, dim):
        if len(self.queue[dim]) == self.w:
            self.queue[dim].pop()
        self.queue[dim].insert(0, abs(residual))

        self.rsum[dim] = sum(self.queue[dim])
        if sum(self.queue[dim]) > self.tao[dim]:
            return True    # alert
        else:
            return False    # no alert

    # residuals: residual data for all dimension in this timestep
    def detect(self, residuals):
        for i in range(self.m):
            self.results[i] = self.detectOneDim(residuals[i], i)
        return self.results

    def alarmOrN(self):
        if sum(self.results) >= 1:
            return True
        else:
            return False
