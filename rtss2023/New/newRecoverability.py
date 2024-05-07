import numpy as np
from utils.formal.zonotope import Zonotope
import time

class Reachability:
    def __init__(self, A, B, U: Zonotope, v_lo, v_up, target_lo, target_up, safe_lo, safe_up, max_step=20):
        self.A = A
        self.B = B
        self.U = U
        self.v_lo = v_lo
        self.v_up = v_up
        self.t_lo = target_lo
        self.t_up = target_up
        self.s_lo = safe_lo
        self.s_up = safe_up
        self.max_step = max_step

        # A^i
        self.A_i = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_i.append(A @ self.A_i[-1])
        # A^i BU
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        # l for support function
        self.l = np.eye(A.shape[0])

        # E
        self.E = self.reach_E()
        self.E_lo, self.E_up = self.bound_E()

        # A^i p
        self.Aip_up, self.Aip_lo = self.bound_Aiv()

        self.ddl = self.max_step
        self.L_lo = []
        self.L_up = []
        self.D_lo = []
        self.D_up = []

    def reach_E(self):
        E = []
        for i in range(self.max_step):
            if i == 0:
                E.append(self.A_i_B_U[0])
            else:
                E.append(E[-1] + self.A_i_B_U[i])
        return E

    def bound_E(self):
        E_lo = []
        E_up = []
        for i in range(self.max_step):
            lo = []
            up = []
            for j in range(self.A.shape[0]):
                lo.append(self.E[i].support(self.l[j], -1))
                up.append(self.E[i].support(self.l[j]))
            E_lo.append(lo)
            E_up.append(up)
        return E_lo, E_up

    def bound_Aiv(self):
        lo = []
        up = []
        for i in range(self.max_step):
            if i == 0:
                lo.append(self.v_lo)
                up.append(self.v_up)
            else:
                lo.append(lo[i - 1] + self.A_i[i] @ self.v_lo)
                up.append(up[i - 1] + self.A_i[i] @ self.v_up)
        return lo, up

    def minkowski_dif1(self, first_lo, first_up, second: Zonotope):
        lo = np.zeros(self.A.shape[0])
        up = np.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            lo[i] = first_lo[i] - second.support(self.l[i], -1)
            up[i] = first_up[i] - second.support(self.l[i])
        return lo, up

    def minkowski_dif(self, first_lo, first_up, second_lo, second_up):
        return first_lo - second_lo, first_up - second_up

    def recoverability(self, x_hat, theta):
        theta_lo = np.array(theta)[:, 0]
        theta_up = np.array(theta)[:, 1]

        self.L_lo = []
        self.L_up = []
        self.D_lo = []
        self.D_up = []
        for i in range(self.max_step):
            self.L_lo.append(self.A_i[i] @ x_hat + self.A_i[i] @ theta_lo + self.Aip_lo[i])
            self.L_up.append(self.A_i[i] @ x_hat + self.A_i[i] @ theta_up + self.Aip_up[i])

            # check ddl
            if not self.inBox(self.L_lo[-1], self.L_up[-1], self.s_lo, self.s_up):
                break

            # check recoverability
            self.D_lo.append(self.t_lo - self.L_lo[-1])
            self.D_up.append(self.t_up - self.L_up[-1])

            # check D empty
            if not self.check_empty(self.D_lo[-1], self.D_up[-1]):
                closest, intersect, iteration = self.check_intersection(self.E[i], self.D_lo[-1], self.D_up[-1])
                self.ddl = i


    def check_intersection(self, E: Zonotope, D_lo, D_up):
        ord = E.g.shape[1]
        intersectCase = 1

        adjustDir = self.emptySet

        return closest, intersect, iteration

    # check first box in second box, 0 not in, 1 in
    def inBox(self, first_lo, first_up, second_lo, second_up):
        for i in range(self.A.shape[0]):
            if first_lo[i] < second_lo[i] or first_up[i] > second_up[i]:
                return 0
        return 1

    def check_empty(self, D_lo, D_up):
        for i in range(self.A.shape[0]):
            if D_lo[i] > D_up[i]:
                return 0
        return 1