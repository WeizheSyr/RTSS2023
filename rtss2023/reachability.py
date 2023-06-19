import numpy as np
from utils.formal.zonotope import Zonotope


class Reachability:
    """
    Analyze the reachable set
    Platoon for test
    """
    def __init__(self, A, B, p: Zonotope, U: Zonotope, target: Zonotope, max_step=20, c=None):
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
        self.A_i_p = [self.A_i_B_U[val] @ self.p for val in range(max_step)]
        self.l = np.eye(A.shape[0])
        self.E_up_s, self.E_low_s = self.reachable_of_E()
        self.D1s = self.D1_support_function()
        self.D4s = self.D4_support_function()

    # control envelope
    # support function
    def reachable_of_E(self):
        E_up_s = []
        E_low_s = []
        for i in range(self.max_step):
            temp = []
            if i == 0:
                for j in range(self.A.shape[0]):
                    temp.append(self.A_i_B_U[0].support(self.l[j]))
            else:
                for j in range(self.A.shape[0]):
                    temp.append(temp[-1] + self.A_i_B_U[i].support(self.l[j]))
            E_up_s.append(temp)
            E_low_s.append(-1 * temp)
        return E_up_s, E_low_s

    # target set
    # support function of D1
    def D1_support_function(self):
        D1s = []
        for i in range(self.max_step):
            temp = []
            for j in range(self.A.shape[0]):
                temp.append(self.targetz.support(self.l[j]))
            D1s.append(temp)
        return D1s

    # support function of D4
    def D4_support_function(self):
        D4s = []
        for i in range(self.max_step):
            temp = []
            if i == 0:
                for j in range(self.A.shape[0]):
                    temp.append((self.A_i[0] @ self.p).support(-1 * self.l[j]))
            else:
                for j in range(self.A.shape[0]):
                    temp.append(temp[-1] + (self.A_i[i] @ self.p).support(-1 * self.l[j]))
            D4s.append(temp)
        return D4s

    # zonotope
    def reachable_of_D23(self, x_tilda: Zonotope, theta: Zonotope):
        D2 = []
        D3 = []
        for i in range(self.max_step):
            D2.append(self.A_i[i] @ x_tilda)
            D3.append(self.A_i_B_U[i] @ theta)
        return D2, D3

    # check recovery-able at timestep d
    # True or False
    def check_recovery(self, d, D2, D3):
        result = True
        D_up_s = []
        D_low_s = []
        # i: l direction
        for i in range(self.A_i_p[0]):
            # D_up_s
            temp1 = self.D1s[d][i]
            temp1 += (-1 * self.l[i]).T * D2[i]
            temp1 += D3[i].support(-1 * self.l[i])
            temp1 += self.D4s[d][i]
            D_up_s.append(temp1)

            # D_low_s
            temp2 = self.D1s[d][i]
            temp2 -= (-1 * self.l[i]).T * D2[i]
            temp2 -= D3[i].support(-1 * self.l[i])
            temp2 -= self.D4s[d][i]
            D_low_s.append(temp2)

            # check intersection
            if self.E_low_s[i] > D_up_s[i] or self.E_up_s[i] < D_low_s:
                result = False
                return result

    # k level recovery-ability
    # ith timestep can't recovery
    def recovery_ability(self, x_tilda: Zonotope, theta: Zonotope):
        D2, D3 = self.reachable_of_D23(x_tilda, theta)
        k = 0
        recover = []
        for d in range(self.max_step):
            recover.append(self.check_recovery(d, D2, D3))

        for i in range(self.max_step):
            if recover[i] == 1:
                k = k + 1
            if k > 0 and recover[i] == 0:
                return k, i
        return k, self.max_step