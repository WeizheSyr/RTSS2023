import numpy as np
from utils.formal.zonotope import Zonotope
import time
from copy import deepcopy

class Reachability:
    def __init__(self, A, B, U: Zonotope, P: Zonotope, v_lo, v_up, target_lo, target_up, safe_lo, safe_up, max_step=20):
        self.A = A
        self.B = B
        self.U = U
        self.P = P
        self.v_lo = v_lo
        self.v_up = v_up
        self.t_lo = target_lo
        self.t_up = target_up
        self.s_lo = safe_lo
        self.s_up = safe_up
        self.max_step = max_step

        # l for support function
        self.l = np.eye(A.shape[0])

        # A^i
        self.A_i = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_i.append(A @ self.A_i[-1])
        self.abs_A_i = deepcopy(self.A_i)
        # for i in range(max_step):
        #     for j in range(self.A.shape[0]):
        #         for k in range(self.A.shape[0]):
        #             self.abs_A_i[i][j][k] = np.abs(self.abs_A_i[i][j][k])
        # self.inv_abs_A_i = self.abs_A_i
        # for i in range(max_step):
        #     self.inv_abs_A_i[i] = np.linalg.inv(self.inv_abs_A_i[i])

        # A^i BU
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        self.A_i_B_U_lo, self.A_i_B_U_up = self.bound_A_i_B_U()

        # E
        self.E = self.reach_E()
        self.E_lo, self.E_up = self.bound_E()

        # A^d \hat{x}_t + A^{d-1} Bu_t
        self.L1 = []

        # A^i p
        self.AiP = [val @ self.P for val in self.A_i]
        self.Aip_lo, self.Aip_up = self.bound_AiP()

        self.ddl = self.max_step
        self.L_lo = [[],[],[]]
        self.L_up = [[],[],[]]
        self.D_lo = [[],[],[]]
        self.D_up = [[],[],[]]

        self.cloest = [[],[],[]]
        self.alpha = [[],[],[]]
        self.intersect = [[], [], []]
        self.empty = [[],[],[]]

        self.k = 3

    def get_bound(self, X: Zonotope):
        lo = []
        up = []
        for i in range(self.A.shape[0]):
            lo.append(X.support(self.l[i], -1))
            up.append(X.support(self.l[i]))
        return lo, up

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

    def bound_A_i_B_U(self):
        A_i_B_U_lo = []
        A_i_B_U_up = []
        for i in range(self.max_step):
            lo = []
            up = []
            for j in range(self.A.shape[0]):
                lo.append(self.A_i_B_U[i].support(self.l[j], -1))
                up.append(self.A_i_B_U[i].support(self.l[j]))
            A_i_B_U_lo.append(lo)
            A_i_B_U_up.append(up)
        return A_i_B_U_lo, A_i_B_U_up

    def bound_AiP(self):
        los = []
        ups = []
        for i in range(self.max_step):
            lo, up = self.get_bound(self.AiP[i])
            los.append(lo)
            ups.append(up)
        return los, ups

    # minkowski difference between box and zonotope
    def minkowski_dif1(self, first_lo, first_up, second: Zonotope):
        lo = np.zeros(self.A.shape[0])
        up = np.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            lo[i] = first_lo[i] - second.support(self.l[i], -1)
            up[i] = first_up[i] - second.support(self.l[i])
        return lo, up

    # minkowski difference between box and box
    def minkowski_dif(self, first_lo, first_up, second_lo, second_up):
        return first_lo - second_lo, first_up - second_up

    def deadline(self, x_hat, thetaz):
        self.L_lo = [[] for i in range(self.k)]
        self.L_up = [[], [], []]
        self.D_lo = [[], [], []]
        self.D_up = [[], [], []]
        self.cloest = [[], [], []]
        self.alpha = [[], [], []]
        self.intersect = [[], [], []]
        self.empty = [[], [], []]
        for j in range(1, 4):
            for d in range(self.max_step - j):
                # sum_{i=1}^{j} A^{j+d-i} BU
                A_B_U_lo = np.zeros(self.A.shape[0])
                A_B_U_up = np.zeros(self.A.shape[0])
                for i in range(j):
                    A_B_U_lo = A_B_U_lo + self.A_i_B_U_lo[j + d - i - 1]
                    A_B_U_up = A_B_U_up + self.A_i_B_U_up[j + d - i - 1]

                theta_lo, theta_up = self.get_bound(self.A_i[j + d] @ thetaz)
                # L_lo = self.A_i[j + d] @ x_hat + theta_lo + A_B_U_lo + self.Aip_lo[j + d - 1]
                # L_up = self.A_i[j + d] @ x_hat + theta_up + A_B_U_up + self.Aip_up[j + d - 1]
                L_lo = self.A_i[j + d] @ x_hat + theta_lo + self.Aip_lo[j + d - 1]
                L_up = self.A_i[j + d] @ x_hat + theta_up + self.Aip_up[j + d - 1]
                if self.inBox(L_lo, L_up, self.s_lo, self.s_up):
                    self.L_lo[j - 1].append(L_lo)
                    self.L_up[j - 1].append(L_up)
                    self.D_lo[j - 1].append(self.t_lo - L_lo)
                    self.D_up[j - 1].append(self.t_up - L_up)
                else:
                    print('unsafe')
                    break

    def recoverable(self, x_hat, theta):
        theta_lo = np.array(theta)[:, 0]
        theta_up = np.array(theta)[:, 1]
        thetaz = Zonotope.from_box(theta_lo, theta_up)
        self.deadline(x_hat, thetaz)
        # j = 1 check recoverability

        # flag = 0
        # for i in range(len(self.L_up[0])):
        #     # check empty
        #     if self.check_empty(self.D_lo[0][i], self.D_up[0][i]):
        #         closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[0][i], self.D_up[0][i])
        #         self.cloest[0].append(closest)
        #         self.alpha[0].append(alpha)
        #         self.intersect[0].append(intersect)
        #         self.empty[0].append(0)
        #         if intersect == 1:
        #             flag = flag + 1
        #     else:
        #         self.empty[0].append(1)

        self.flag = [0, 0, 0]
        for j in range(3):
            for i in range(len(self.L_up[j])):
                # check empty
                if self.check_empty(self.D_lo[j][i], self.D_up[j][i]):
                    closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[j][i], self.D_up[j][i])
                    self.cloest[j].append(closest)
                    self.alpha[j].append(alpha)
                    self.intersect[j].append(intersect)
                    self.empty[j].append(0)
                    if intersect == 1:
                        self.flag[j] = self.flag[j] + 1
                else:
                    closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[j][i], self.D_up[j][i])
                    self.cloest[j].append(closest)
                    self.alpha[j].append(alpha)
                    self.intersect[j].append(0)
                    self.empty[j].append(1)
        return self.flag

    def threshold_decrease(self, tau):
        # increase recoverable time window
        j = 1
        # delta_tau = np.ones(self.A.shape[0]) * tau * 0.05
        delta_tau = np.ones(self.A.shape[0]) * 0.001
        stop = 0
        while(True):
            for i in range(len(self.L_up[j])):
                self.D_lo[j][i] = self.D_lo[j][i] - self.abs_A_i[j + i] @ delta_tau
                self.D_up[j][i] = self.D_up[j][i] + self.abs_A_i[j + i] @ delta_tau
                closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[j][i], self.D_up[j][i])
                if intersect == 1:
                # if self.point_in_box(self.cloest[j][i], self.D_lo[j][i], self.D_up[j][i]):
                    stop = 1
                    break
            if stop == 1:
                break
            else:
                delta_tau = delta_tau + 0.001
        return delta_tau

    def threshold_increase(self, tau):
        j = 2
        delta_tau = np.ones(self.A.shape[0]) * 0.001
        stop = 1
        while(True):
            # for i in range(len(self.L_up[j])):
            for i in range(np.sum(self.empty[j])):
                self.D_lo[j][i] = self.D_lo[j][i] + self.abs_A_i[j + i] @ delta_tau
                self.D_up[j][i] = self.D_up[j][i] - self.abs_A_i[j + i] @ delta_tau
                closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[j][i], self.D_up[j][i])
                if intersect == 1:
                # if self.point_in_box(self.cloest[j][i], self.D_lo[j][i], self.D_up[j][i]):
                    stop = 0
            if stop == 1:
                break
            else:
                delta_tau = delta_tau + 0.001
        return delta_tau

    def threshold_increase1(self, tau):
        j = 2
        delta_tau = np.ones(self.A.shape[0]) * 0.0006
        stop = 1
        while(True):
            # for i in range(len(self.L_up[j])):
            for i in range(np.sum(self.empty[j])):
                self.D_lo[j][i] = self.D_lo[j][i] + self.abs_A_i[j + i] @ delta_tau
                self.D_up[j][i] = self.D_up[j][i] - self.abs_A_i[j + i] @ delta_tau
                closest, alpha, intersect = self.check_intersection(self.E[i], self.D_lo[j][i], self.D_up[j][i])
                if intersect == 1:
                # if self.point_in_box(self.cloest[j][i], self.D_lo[j][i], self.D_up[j][i]):
                    stop = 0
            if stop == 1:
                break
            else:
                delta_tau = delta_tau + 0.0006
        return delta_tau

    # j = 1
    # def recoverability1(self, x_hat, theta):
    #     theta_lo = np.array(theta)[:, 0]
    #     theta_up = np.array(theta)[:, 1]
    #
    #     self.L_lo = []
    #     self.L_up = []
    #     self.D_lo = []
    #     self.D_up = []
    #     self.cloest = []
    #     self.intersect = []
    #     self.empty = []
    #     for i in range(1, self.max_step):
    #         self.L_lo.append(self.A_i[i] @ x_hat + self.A_i[i] @ theta_lo + self.Aip_lo[i-1] + self.A_i[i-1] @ self.B @ ut)
    #         self.L_up.append(self.A_i[i] @ x_hat + self.A_i[i] @ theta_up + self.Aip_up[i-1] + self.A_i[i-1] @ self.B @ ut)
    #         # check ddl
    #         if not self.inBox(self.L_lo[-1], self.L_up[-1], self.s_lo, self.s_up):
    #             self.ddl = i - 1
    #             print("ddl", self.ddl)
    #             break
    #
    #         # check recoverability
    #         self.D_lo.append(self.t_lo - self.L_lo[-1])
    #         self.D_up.append(self.t_up - self.L_up[-1])
    #
    #         # check D empty
    #         if self.check_empty(self.D_lo[-1], self.D_up[-1]):
    #             # check intersection between D and E
    #             closest, intersect = self.check_intersection(self.E[i], self.D_lo[-1], self.D_up[-1])
    #             self.cloest.append(closest)
    #             self.intersect.append(intersect)
    #             self.empty.append(0)
    #         else:
    #             # D is empty, no need to check more
    #             self.empty.append(1)
    #             return len(self.intersect)
    #     return len(self.intersect)

    def check_intersection(self, E: Zonotope, D_lo, D_up):
        ord = E.g.shape[1]
        stop = 1
        flag = 0

        center = (D_up + D_lo) / 2
        start = E.c
        dir = np.zeros(ord)
        iteration = 0
        i = 0
        while i < ord:
            move = 0
            t = self.get_t(center - start, E.g[:, i])
            if t + dir[i] >= 1:
                move = 1 - dir[i]
                dir[i] = 1
            elif t + dir[i] <= -1:
                move = -1 - dir[i]
                dir[i] = -1
            else:
                move = t
                dir[i] = move + dir[i]
            if not self.newEqual(move, 0):
                stop = 0
            start = start + move * E.g[:, i]
            if self.point_in_box(start, D_lo, D_up) and flag == 0:
                flag = 1

            if i == ord - 1 and stop == 0:
                iteration = iteration + 1
                if iteration < 20:
                    i = -1
                # i = i
                stop = 1
            i = i + 1
        return start, dir, flag

    # check first box in second box, 0 not in, 1 in
    def inBox(self, first_lo, first_up, second_lo, second_up):
        for i in range(self.A.shape[0]):
            if first_lo[i] < second_lo[i] or first_up[i] > second_up[i]:
                return 0
        return 1

    def point_in_box(self, point, box_lo, box_up):
        for i in range(self.A.shape[0]):
            if box_lo[i] > point[i] or box_up[i] < point[i]:
                return 0
        return 1

    def check_empty(self, D_lo, D_up):
        for i in range(self.A.shape[0]):
            if D_lo[i] > D_up[i]:
                return 0
        return 1

    def get_t(self, a, b):
        # a is E*
        # b is generator
        t = np.dot(a, b) / np.dot(b, b)
        return t

    def newEqual(self, a, b):
        if -0.001 < a - b < 0.001:
            return 1
        else:
            return 0