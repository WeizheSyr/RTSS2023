import numpy as np
from utils.formal.zonotope import Zonotope
import time

class Reachability:
    def __init__(self, A, B, P: Zonotope, U: Zonotope, target_low, target_up, max_step=10):
        self.A = A
        self.B = B
        self.P = P
        self.U = U
        # D1
        self.t_lo = target_low
        self.t_up = target_up
        self.max_step = max_step

        # A^i
        self.A_i = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_i.append(A @ self.A_i[-1])
        # A^i BU
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        # A^i P
        self.A_i_P = [self.A_i[val] @ self.P for val in range(max_step)]
        # l for support function
        self.l = np.eye(A.shape[0])

        # E
        self.E = self.reach_E()
        self.E_lo, self.E_up = self.bound_E()

        # D
        self.D2 = None
        self.D3 = None
        self.D4 = self.reach_D4()
        self.D_lo = None
        self.D_up = None

        # result of checking intersection
        self.intersection = np.zeros(self.max_step)
        self.delta_theta = np.zeros(A.shape[0])

    # zonotope E
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

    # zonotope D4
    def reach_D4(self):
        D4 = []
        for i in range(self.max_step):
            if i == 0:
                D4.append(self.P)
            else:
                D4.append(D4[-1] + self.A_i[i - 1] @ self.P)
        return D4

    # update D2, D3
    def reach_D23(self, x_hat, theta):
        D2 = []
        D3 = []
        for i in range(self.max_step):
            D2.append(self.A_i[i] @ x_hat)
            D3.append(self.A_i[i] @ theta)
        return D2, D3

    # bound of box D
    def D_bound(self):
        D_lo = []
        D_up = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            lo, up= self.minkowski_dif(self.t_lo, self.t_up, second)
            D_lo.append(lo)
            D_up.append(up)
        return D_lo, D_up

    # minkowski_dif between box and zonotope
    def minkowski_dif(self, first_lo, first_up, second: Zonotope):
        lo = np.zeros(self.A.shape[0])
        up = np.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            lo[i] = first_lo[i] - second.support(self.l[i], -1)
            up[i] = first_up[i] - second.support(self.l[i])
        return lo, up

    # calculate k level recoverability of the current system
    def k_level(self, x_hat, theta):
        self.D2, self.D3 = self.reach_D23(x_hat, theta)
        self.D_lo, self.D_up = self.D_bound()

        ks = np.zeros(self.max_step)
        distance = np.zeros(self.max_step)
        for i in range(self.max_step):
            ks[i], distance[i] = self.check_intersection(i)
            print("i", i, "ks[i]", ks[i])

        k = np.sum(ks)
        if k == 0:
            return 0, 0, 0

        start = 0
        end = 0
        for i in range(self.max_step):
            if ks[i] == 1 and start == 0:
                start = i
            if ks[i] == 0 and start == 1:
                end = i - 1
            if i == self.max_step - 1 and end == 0:
                end = self.max_step - 1

        return k, start, end

    # check dth step's intersection
    # return 1/0: intersect or not, distance: closest point's distance
    def check_intersection(self, d):
        ord = self.E[d].g.shape[1]
        dim = self.A.shape[0]

        # precheck before explore
        if self.preCheck(self.D_lo[d], self.D_up[d], d):
            return 0, np.zeros(dim)
        new_lo, new_up = self.cropBox(self.D_lo[d], self.D_up[d], d)

        start = self.E[d].c
        center = (new_lo + new_up) / 2
        dir = np.zeros(ord)
        move = np.zeros(ord)
        usedout = 1
        i = 0
        iteration = 0

        while i < ord:
            # t = getT(closestPoint(new_low, new_up, start) - start, temp.g[:, i])
            t = self.getT(center - start, self.E[d].g[:, i])
            if -0.00001 <= t <= 0.00001:
                t = 0
            # t = getT(box.c - start, temp.g[:, i])
            # print("t", t)
            if t != 0:
                usedout = 0
                if t + dir[i] >= 1:
                    move[i] = 1 - dir[i]
                    dir[i] = 1
                elif t + dir[i] <= -1:
                    move[i] = -1 - dir[i]
                    dir[i] = -1
                elif -1 <= t + dir[i] <= 1:
                    move[i] = t
                    dir[i] = t + dir[i]
                # print("dir", i, dir[i])
                # print("move", move[i])
                next = start + move[i] * self.E[d].g[:, i]
                # print(next)
                # distance = np.linalg.norm(next - closestPoint(new_low, new_up, next))
                # print("distance", distance)
                re = self.checkinBox(new_lo, new_up, next)
                if re == 1:
                    # print("intersect point", next)
                    return 1, np.zeros(dim)
                # f_low, f_up, signal = checkPass(start, next, box.c, boxG)
                # if signal != -1:
                #     # print("pass intersection", start, next)
                #     break
                start = next

            if i == ord - 1:
                # print("one iteration")
                iteration += 1
                if usedout == 0:
                    i = -1
                    usedout = 1
                else:
                    break
            i += 1
        return 0, np.zeros(dim)

    # projection of the line to the generator
    def getT(self, a, b):
        # b is generator
        k = np.dot(a, b) / np.dot(b, b)
        return k

    # the closest point in the box to the current point
    def closestPoint(self, low, up, point):
        re = np.zeros(low.shape[0])
        for i in range(low.shape[0]):
            if low[i] <= point[i] <= up[i]:
                re[i] = point[i]
            elif point[i] < low[i]:
                re[i] = low[i]
            elif point[i] > up[i]:
                re[i] = up[i]
        return re

    # check a point in the box
    def checkinBox(self, low, up, point):
        result = 1
        for i in range(low.shape[0]):
            if point[i] <= low[i] - 0.01:
                result = 0
                return result
            if point[i] >= up[i] + 0.01:
                result = 0
                return result
        return result

    def checkEmpty(self, lo, up):
        for i in range(self.A.shape[0]):
            if lo[i] >= up[i]:
                return 1
        return 0

    def preCheck(self, lo, up, d):
        for i in range(self.A.shape[0]):
            # empty set
            if lo[i] >= up[i]:
                return 1
            # impossible intersection
            if lo[i] >= self.D_up[d][i] or up[i] <= self.D_lo[d][i]:
                return 1
        return 0

    def cropBox(self, D_lo, D_up, d):
        new_lo = []
        new_up = []
        for i in range(self.A.shape[0]):
            if self.E_lo[d][i] <= D_lo[i] and self.E_up[d][i] >= D_up[i]:
                new_up.append(D_up[i])
                new_lo.append(D_lo[i])
            elif self.E_lo[d][i] >= D_lo[i] and self.E_up[d][i] <= D_up[i]:
                new_up.append(self.E_up[d][i])
                new_lo.append(self.E_lo[d][i])
            elif self.E_lo[d][i] <= D_lo[i] and self.E_up[d][i] <= D_up[i]:
                new_up.append(self.E_up[d][i])
                new_lo.append(D_lo[i])
            elif self.E_lo[d][i] >= D_lo[i] and self.E_up[d][i] >= D_up[i]:
                new_up.append(D_up[i])
                new_lo.append(self.E_lo[d][i])

        new_lo = np.array(new_lo)
        new_up = np.array(new_up)
        return new_lo, new_up
