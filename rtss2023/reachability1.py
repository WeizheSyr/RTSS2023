import numpy as np
from utils.formal.zonotope import Zonotope

class Reachability1:
    def __init__(self, A, B, p:Zonotope, U:Zonotope, target:Zonotope, target_low, target_up, max_step=40, c=None):
        self.A = A
        self.B = B
        self.p = p
        self.Uz = U
        self.targetz = target
        self.target_low = target_low
        self.target_up = target_up
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
                E.append(E[-1] + self.A_i_B_U[i])
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
        D_low = []
        D_up = []
        length = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            first = self.D1[i]
            low, up = self.minkowski_dif(first, second)
            D_low.append(low)
            D_up.append(up)
            len = []
            for j in range(self.A.shape[0]):
                len.append(up[j] - low[j])
            length.append(len)
        return D_low, D_up

    def minkowski_dif(self, first:Zonotope, second:Zonotope):
        D_low = []
        D_up = []
        for i in range(self.A.shape[0]):
            # D_low
            temp1 = first.support(self.l[i], -1)
            temp2 = second.support(self.l[i], -1)
            D_low.append(self.target_low[i] - temp2)

            # D_up
            temp3 = first.support(self.l[i])
            temp4 = second.support(self.l[i])
            D_up.append(self.target_up[i] - temp4)

        return D_low, D_up

    # def intesection(self, first:Zonotope, second:Zonotope):
    #     result = []
    #     dist = []
    #     for i in range(first.g.shape[0]):
    #         first_up = first.c + first.g[i]
    #         first_low = first.c - first.g[i]
    #         first_up, first_low = self.check(first_up, first_low)
    #
    #         second_up = second.c + second.g[i]
    #         second_low = second.c - second.g[i]
    #         second_up, second_low = self.check(second_up, second_low)
    #
    #         if first_low > second_up:
    #     return


    def check(self, first, second):
        if first < second:
            return second, first
        else:
            return first, second

    def recovery_ability(self, x_hat:Zonotope, theta:Zonotope):
        self.D2, self.D3 = self.reach_D23(x_hat, theta)
        result = self.D()
        return result