import numpy as np
import math
from copy import deepcopy

class Estimator:
    def __init__(self, Ad, Bd, x_hat, uup=1, ulow=0, max_k=50, p=0.0001, v=0.0001, sup=2.5, slow=1.5, c=0, k=5):
        self.p = p
        self.x_hat = x_hat
        self.max_k = max_k
        self.uup = uup
        self.ulow = ulow
        self.sup = sup
        self.slow = slow
        self.Ad = Ad
        self.Bd = Bd
        self.Ad_k_Bd = []
        # self.n = self.Ad.shape[0]
        self.n = 1
        self.Ad_k = [np.eye(self.n)]
        self.c = c
        self.Ad_x = self.Ad.dot(self.x_hat)
        self.Ad_up = self.Ad.dot(self.up)
        self.Ad_low = self.Ad.dot(self.low)
        for i in range(max_k):
            self.Ad_k.append(self.Ad_k[-1].dot(self.Ad))
        self.Ad_k_Bd = [i.dot(self.Bd) for i in self.Ad_k]
        self.Ad_k_c = [i.dot(self.c) for i in self.Ad_k]
        self.Ad_k_p = [i.dot(self.p) for i in self.Ad_k]
        self.theta = 0
        self.actmax = 0
        self.k = k

    # E low < s up
    def reachability(self, up, low):
        if abs(up) > abs(low):
            self.theta = abs(up)
        else:
            self.theta = abs(low)
        control_up = []
        control_low = []
        control_sum_term_up = self.uup
        control_sum_term_low = self.ulow
        for j in range(self.max_k):
            control_sum_term_up += self.Ad_k_Bd[j] @ self.uup
            control_up.append(control_sum_term_up)
            control_sum_term_low += self.Ad_k_Bd[j] @ self.ulow
            control_low.append(control_sum_term_low)

        safeset_up = []
        safeset_low = []
        safeset_sum_term_up = self.sup
        safeset_sum_term_low = self.slow
        for j in range(self.max_k):
            safeset_sum_term_up = safeset_sum_term_up - self.Ad_k_c[j] - self.Ad_k_p[j] - self.Ad_k[j].dot(self.x_hat) - self.Ad_k[j].dot(self.theta)
            safeset_up.append(safeset_sum_term_up)
            if safeset_sum_term_up <= 0:
                self.actmax = j
                break
            safeset_sum_term_low = safeset_sum_term_low - self.Ad_k_c[j] - self.Ad_k_p[j] - self.Ad_k[j].dot(self.x_hat) - self.Ad_k[j].dot(self.theta)
            safeset_low.append(safeset_sum_term_low)

        level = []
        for j in range(self.actmax):
            if control_low[j] <= safeset_up[j]:
                level.append(j)

        if len(level) < self.k:
            temp = level[-1] + 1
            new = control_low[temp]
        elif len(level) > self.k:
            temp = level[k]
            new = control_low[temp] - safeset_up[temp]

        return len(level), new

    def adjust(self, new):
        if new > 0: # reduce
            return
        else:   # enlarge
            return


if __name__ == '__main__':
    est = Estimator(Ad= 2,Bd=3,max_k=3)
    print(est.Ad_k)