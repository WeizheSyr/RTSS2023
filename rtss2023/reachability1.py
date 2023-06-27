import numpy as np
from utils.formal.zonotope import Zonotope

class Reachability1:
    def __init__(self, A, B, p:Zonotope, U:Zonotope, target:Zonotope, max_step=40, c=None):
        self.A = A
        self.B = B
        self.p = p
        self.Uz = U
        self.targetz = target
        self.max_step = max_step
        self.c = c
        self.A_i = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_i.append(A @ self.A_i[-1])
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        self.A_i_p = [self.A_i[val] @ self.p for val in range(max_step)]
        self.l = np.eye(A.shape[0])
        self.E = self.reach_E()
        self.D1 = self.reach_D1()
        self.D2 = None
        self.D3 = None
        self.D4 = self.reach_D4()

    def reach_E(self):
        E = []
        for i in range(self.max_step):
            if i == 0:
                E.append(self.A_i_B_U[0])
            else:
                E.append(self.E[-1] + self.A_i_B_U[i])
        return E

    def reach_D1(self):
        D1 = []
        for i in range(self.max_step):
            D1.append(self.targetz)
        return D1

    def reach_D4(self):
        D4 = []
        for i in range(self.max_step):
            if i == 0:
                D4.append(self.p)
            else:
                D4.append(D4[-1] + self.A_i[i - 1] @ self.p)
        return D4

    def reach_D23(self, x_hat, theta):
        D2 = []
        D3 = []
        for i in range(self.max_step):
            D2.append(self.A_i[i] @ x_hat)
            D3.append(self.A_i[i] @ theta)
        return D2, D3

    def D(self):
        result = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            first = self.D1[i]
            result.append(self.minkowski_dif(first, second))
        return result

    def minkowski_dif(self, first:Zonotope, second:Zonotope):
        c = first.c
        g = np.empty(first.g.shape[0])
        for i in range(g.shape[0]):
            g = first.g[i] - second.g[i]
        result = Zonotope(c, g)
        return result

    def intesection(self, first:Zonotope, second:Zonotope):
        result = []
        dist = []
        for i in range(first.g.shape[0]):
            first_up = first.c + first.g[i]
            first_low = first.c - first.g[i]
            first_up, first_low = self.check(first_up, first_low)

            second_up = second.c + second.g[i]
            second_low = second.c - second.g[i]
            second_up, second_low = self.check(second_up, second_low)

            if first_low > second_up:
        return


    def check(self, first, second):
        if first < second:
            return second, first
        else:
            return first, second