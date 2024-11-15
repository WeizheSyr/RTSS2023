import numpy as np
from utils.formal.zonotope import Zonotope

import time

class Reachability1:
    def __init__(self, A, B, p: Zonotope, U: Zonotope, target: Zonotope, target_low, target_up, max_step=20, c=None):
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
        self.inv_A_d = self.inv_A_d()
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        self.A_i_p = [self.A_i[val] @ self.p for val in range(max_step)]
        self.l = np.eye(A.shape[0])

        self.E = self.reach_E()
        self.E_low, self.E_up = self.E_bound()

        self.D1 = self.reach_D1()
        self.D2 = None
        self.D3 = None
        self.D4 = self.reach_D4()
        self.D_low = None
        self.D_up = None

        self.result = None
        self.intersection = None
        self.distance = None

        self.delta_theta = None

    def inv_A_d(self):
        inv_A_d = []
        for i in range(self.max_step):
            inv_A_d.append(np.linalg.inv(self.A_i[i]))
        return inv_A_d

    def reach_E(self):
        E = []
        for i in range(self.max_step):
            if i == 0:
                E.append(self.A_i_B_U[0])
            else:
                E.append(E[-1] + self.A_i_B_U[i])
        return E

    def E_bound(self):
        E_low = []
        E_up = []
        for i in range(self.max_step):
            low = []
            up = []
            for j in range(self.A.shape[0]):
                low.append(self.E[i].support(self.l[j], -1))
                up.append(self.E[i].support(self.l[j]))
            E_low.append(low)
            E_up.append(up)
        return E_low, E_up

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

    def D_bound(self):
        start = time.time()
        D_low = []
        D_up = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            first = self.D1[i]
            low, up = self.minkowski_dif(first, second)
            D_low.append(low)
            D_up.append(up)
        end = time.time()
        print("D_bound", end - start)
        return D_low, D_up

    def D_bound_box(self):
        D_boxes = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            first = self.D1[i]
            low, up = self.minkowski_dif(first, second)
            for j in range(self.A.shape[0]):
                if low[j] >= up[j]:
                    print("empty set at ", i)
                    continue
            D_boxes.append(Zonotope.from_box(np.array(low), np.array(up)))
        return D_boxes

    def minkowski_dif(self, first: Zonotope, second: Zonotope):
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

    # check the intersection of D and E for one time step
    def check_intersection(self, i):
        inter = []
        dis = []
        # print("check intersection", i)
        # print("D_low[i]", self.D_low[i])
        # print("D_up[i]", self.D_up[i])
        # print("E_low[i]", self.E_low[i])
        # print("E_up[i]", self.E_up[i])
        for j in range(self.A.shape[0]):
            if self.D_low[i][j] > self.D_up[i][j]:
                # print("self.D_low[i][j] > self.D_up[i][j]", self.D_low[i][j], self.D_up[i][j])
                # D is empty set
                inter.append(0)
            # elif self.E_up[i][j] >= self.D_low[i][j] and self.E_low[i][j] <= self.D_up[i][j]:
            #     # intersection E lower than D
            #     inter.append(1)
            elif self.E_up[i][j] < self.D_low[i][j]:
                # no intersection E lower than D
                inter.append(-1)
            # elif self.E_low[i][j] <= self.D_up[i][j] and self.E_up[i][j] >= self.D_low[i][j]:
            #     # intersection E higher than D
            #     inter.append(1)
            elif self.E_low[i][j] > self.D_up[i][j]:
                # no intersection E higher than D
                inter.append(-1)
            # elif self.E_low[i][j] > self.D_low[i][j] and self.E_up < self.D_up:
            #     inter.append(1)
            else:
                inter.append(1)

            dis.append(self.D_up[i][j] - self.D_low[i][j])
        return inter, dis

    # check the recoverability in later max steps
    def recoverability(self, x_hat: Zonotope, theta: Zonotope):
        result = []
        intersection = []
        distance = []
        self.D2, self.D3 = self.reach_D23(x_hat, theta)
        self.D_low, self.D_up = self.D_bound()
        for i in range(self.max_step):
            inter, dis = self.check_intersection(i)
            # print(i)
            # print("inter", inter)
            intersection.append(inter)
            # distance.append(dis)

        # check recoverability
        for i in range(self.max_step):
            t = 1
            for j in range(self.A.shape[0]):
                if intersection[i][j] != 1:
                    t = 0
            result.append(t)
        self.result = result
        self.intersection = intersection
        self.distance = distance

    def k_level(self, x_hat: Zonotope, theta: Zonotope):
        self.recoverability(x_hat, theta)

        startTimeBox = time.time()
        # opt intersection
        D_boxes = self.D_bound_box()
        endTimeBox = time.time()
        print("timeBox", endTimeBox - startTimeBox)

        print("results", self.result)
        k1 = 0
        start1 = -1
        end1 = -1
        k2 = 0
        start2 = -1
        end2 = -1
        for i in range(self.max_step):
            if self.result[i] == 1:
                if k1 == 0:
                    start1 = i
                    k1 = k1 + 1
                elif k1 != 0 and end1 == -1:
                    k1 = k1 + 1
                elif k1 != 0 and k2 == 0:
                    start2 = i
                    k2 = k2 + 1
                elif k2 != 0:
                    k2 = k2 + 1
            elif self.result[i] == 0:
                # last intersect
                if k1 != 0 and end1 == -1:
                    end1 = i - 1
                elif k2 != 0 and end2 == -1:
                    end2 = i - 1

        if k1 == 1 and k2 != 0:
            if end2 == -1:
                end2 = self.max_step - 1
            return k2, start2, end2
        else:
            if end1 == -1 and k1 != 0:
                end1 = self.max_step - 1
            return k1, start1, end1

        # for i in range(self.max_step):
        #     if self.result[i] == 1:
        #         if k == 0:
        #             start = i
        #         k = k + 1
        #     elif k > 0 and self.result[i] == 0:
        #         # intersect
        #         end = i - 1
        #         return k, start, end
        # if k == 0:
        #     # not intersect at all
        #     return k, start, end
        # else:
        #     # intersect and end at finally
        #     return k, start, self.max_step - 1

    def adjust_new(self, k, start, end, klevel):
        delta_theta = np.zeros(self.A.shape[0])
        if k < klevel:
            # print("k, start, end", k, start, end)
            # print("self.result", self.result)
            # decrease theta
            if end == -1 and start == -1:
                # not intersection at all
                isempty = -1

                # search for the empty dimension
                # D become empty before intersect
                for i in range(self.max_step):
                    for j in range(self.A.shape[0]):
                        if self.D_low[i][j] >= self.D_up[i][j]:
                            isempty = i
                            delta_theta[j] = -0.1
                            # print("D become empty before intersect")
                    if isempty != -1:
                        break

                # D does not become empty before max_step
                if isempty == -1:
                    for j in range(self.A.shape[0]):
                        if self.D_low[self.max_step - 1][j] >= self.E_up[self.max_step - 1][j] or \
                                self.D_up[self.max_step - 1][j] <= self.E_low[self.max_step -1][j]:
                            delta_theta[j] = -0.1
                            # print("D does not become empty before max_step")

            elif end == self.max_step - 1 and start > -1:
                # intersect and D does not become empty before max_step
                for j in range(self.A.shape[0]):
                    if self.D_low[start - 1][j] >= self.E_up[start - 1][j] or \
                            self.D_up[start - 1][j] <= self.E_low[start - 1][j]:
                        delta_theta[j] = -0.1
                        # print("intersect and D does not become empty before max_step")
            elif end > -1 and start > -1:
                # intersect and D become empty before max_step
                # search for the empty dimension
                flag_delta = 0
                for j in range(self.A.shape[0]):
                    if self.D_low[end + 1][j] >= self.D_up[end + 1][j]:
                        delta_theta[j] = -0.1
                        flag_delta = 1
                        # print("# intersect and D become empty before max_step")
                if flag_delta == 0:
                    # print("unexpect situation")
                    for j in range(self.A.shape[0]):
                        delta_theta[j] = -0.1
            # else:
                # print("unexpect situation")

        else:
            # increase theta
            for j in range(self.A.shape[0]):
                delta_theta[j] = 0.1

        self.delta_theta = delta_theta
        return delta_theta


    def adjust(self, k, start, end, klevel):
        delta_theta = np.zeros(self.A.shape[0])
        if k < klevel:
            # increase k
            # not intersection at all
            if end == 0:
                isempty = 0
                emptydim = np.zeros(self.A.shape[0])
                for i in range(self.max_step):
                    if isempty == 0:
                        for j in range(self.A.shape[0]):
                            if self.D_low[i][j] >= self.D_up[i][j]:
                                isempty = 1
                                emptydim[j] = 1
                for j in range(self.A.shape[0]):
                    if emptydim[j] == 1:
                        delta_theta[j] = -0.2
                    else:
                        delta_theta[j] = 0

                if isempty == 0:
                    for j in range(self.A.shape[0]):
                        if self.D_low[i][j] >= self.E_up[i][j] or self.D_low[i][j] >= self.E_up[i][j]:
                            delta_theta[j] = -0.2

                flag = 0
                for j in range(self.A.shape[0]):
                    if delta_theta[j] != 0:
                        flag = 1
                if flag == 0:
                    for j in range(self.A.shape[0]):
                        delta_theta[j] = -0.2

                return delta_theta
            else:
                i = start - 1
                for j in range(self.A.shape[0]):
                    dist = self.E_up[i][j] - self.D_low[i][j]
                    dist = 2 * dist / self.distance[i][j]
                    # if dist <= -1 or dist > 0:
                    #     dist = 0
                    if dist > 0:
                        dist = 0
                    else:
                        dist = -0.1
                    delta_theta[j] = dist

        else:
            # decrease k
            for j in range(self.A.shape[0]):
                delta_theta[j] = 0.1
            # i = start + 1
            # for j in range(self.A.shape[0]):
            #     dist = self.E_up[i][j] - self.D_low[i][j]
            #     dist = 2 * dist / self.distance[i][j]
            #     if dist < 0:
            #         dist = 0
            #     else:
            #         dist = 0.1
            #     # if dist >= 1 or dist < 0:
            #     #     dist = 0
            #     delta_theta[j] = dist

        self.delta_theta = delta_theta
        return delta_theta


    # A is a zonotope with generator
    # B is a box
    def attemp_reach(self, A: Zonotope, B):
        p = A.g.shape[1]
