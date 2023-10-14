import numpy as np
from utils.formal.zonotope import Zonotope
import time

class Reachability:
    def __init__(self, A, B, P: Zonotope, U: Zonotope, target_low, target_up, max_step=20):
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

        # adjust Tau
        self.adjustDirs = None
        # s norm positive or negative
        self.sNorm = None
        self.deltaCs = None
        self.deltaEs = None
        self.inOrOuts = None     # 0: out, 1: in
        self.scopes = None
        self.detector = None
        self.emptySet = np.zeros(self.max_step)

        # time
        self.timeIntersection = 0
        self.numIntersection = 0
        self.timeAdjust = 0
        self.numAdjust = 0

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
        sNorm = []
        for i in range(self.max_step):
            second = self.D2[i] + self.D3[i] + self.D4[i]
            sNorm.append(self.getSNorm(self.D3[i]))
            lo, up = self.minkowski_dif(self.t_lo, self.t_up, second)
            D_lo.append(lo)
            D_up.append(up)
        self.sNorm = sNorm
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
        adjustDirs = []
        scopes = []
        inOrOuts = []
        iterations = []
        startTime = time.time()
        self.emptySet = np.zeros(self.max_step)
        for i in range(self.max_step):
            # ks[i], adjustDir, iteration = self.check_intersection(i)
            ks[i], adjustDir, inOrOut, scope, iteration = self.check_intersectionNew(i)
            adjustDirs.append(adjustDir)
            inOrOuts.append(inOrOut)
            scopes.append(scope)
            iterations.append(iteration)
            # print("i", i, "ks[i]", ks[i])
        self.adjustDirs = adjustDirs
        self.inOrOuts = inOrOuts
        self.scopes = scopes
        endTime = time.time()
        print("++++++++++++++++++++++++++++++++")
        print("check intersection time: ",endTime - startTime)
        # self.timeIntersection += endTime - startTime
        # self.numIntersection += 1
        print("ks", ks)
        # print("iterations", iterations)
        # print("avg time", self.timeIntersection / self.numIntersection)
        print("++++++++++++++++++++++++++++++++")

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

        # pre-check before explore
        if self.preCheck(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d]):
            # return 0, np.zeros(self.A.shape[0]), 0
            return 0, np.zeros(self.A.shape[0]), np.zeros(self.A.shape[0]), np.zeros(self.A.shape[0]), 0
        new_lo, new_up = self.cropBox(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d])

        start = self.E[d].c
        next = start
        center = (new_lo + new_up) / 2
        dir = np.zeros(ord)
        move = np.zeros(ord)
        usedout = 1
        i = 0
        iteration = 0

        while i < ord:
            # t = self.getT(self.closestPoint(new_lo, new_up, start) - start, self.E[d].g[:, i])
            t = self.getT(center - start, self.E[d].g[:, i])
            # t = getT(box.c - start, temp.g[:, i])
            # print("t", t)
            if not self.newEqual(t, 0):
                # usedout = 0
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
                if not self.newEqual(move[i], 0):
                    usedout = 0
                next = start + move[i] * self.E[d].g[:, i]
                # print(next)
                # distance = np.linalg.norm(next - closestPoint(new_low, new_up, next))
                # print("distance", distance)
                re = self.checkinBox(new_lo, new_up, next)
                if re == 1:
                    # print("intersect point", next)
                    # return 1, self.adjustDir(new_lo, new_up, next), iteration
                    adjustDir, inOrOut, scope = self.adjustDirNew(new_lo, new_up, next)
                    return 1, adjustDir, inOrOut, scope, iteration
                # f_low, f_up, signal = checkPass(start, next, box.c, boxG)
                # if signal != -1:
                #     # print("pass intersection", start, next)
                #     break
                start = next

            if i == ord - 1:
                # print("one iteration")
                iteration += 1
                if iteration >= 50:
                    break
                if usedout == 0:
                    i = -1
                    usedout = 1
                else:
                    break
            i += 1
        # return 0, self.adjustDir(new_lo, new_up, next), iteration
        adjustDir, inOrOut, scope = self.adjustDirNew(new_lo, new_up, next)
        return 0, adjustDir, inOrOut, scope, iteration

    def check_intersectionNew(self, d):
        ord = self.E[d].g.shape[1]

        t = self.preCheck(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d])
        # pre-check before explore
        if t == -1:
            adjustDir = self.emtpySetDir(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d])
            self.emptySet[d] = 1
            # return 0, adjustDir, inOrOut, scope, 0
            return 0, adjustDir, np.zeros(self.A.shape[0]), np.zeros(self.A.shape[0]), 0
        elif t == 1:
            adjustDir, inOrOut, scope = self.adjustDirsNN(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d])
            return 0, adjustDir, inOrOut, scope, 0
        new_lo, new_up = self.cropBox(self.D_lo[d], self.D_up[d], self.E_lo[d], self.E_up[d])

        start = self.E[d].c
        next = start
        center = (new_lo + new_up) / 2
        dir = np.zeros(ord)
        move = np.zeros(ord)
        usedout = 1
        i = 0
        iteration = 0
        result = 0

        while i < ord:
            # t = self.getT(self.closestPoint(new_lo, new_up, start) - start, self.E[d].g[:, i])
            t = self.getT(center - start, self.E[d].g[:, i])
            # t = getT(box.c - start, temp.g[:, i])
            # print("t", t)
            if not self.newEqual(t, 0):
                # usedout = 0
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
                if not self.newEqual(move[i], 0):
                    usedout = 0
                next = start + move[i] * self.E[d].g[:, i]
                # print(next)
                # distance = np.linalg.norm(next - closestPoint(new_low, new_up, next))
                # print("distance", distance)
                re = self.checkinBox(new_lo, new_up, next)
                if re == 1:
                    # print("intersect point", next)
                    # return 1, self.adjustDir(new_lo, new_up, next), iteration
                    result = 1
                    # adjustDir, inOrOut, scope = self.adjustDirNew(new_lo, new_up, next)
                    # return 1, adjustDir, inOrOut, scope, iteration
                # f_low, f_up, signal = checkPass(start, next, box.c, boxG)
                # if signal != -1:
                #     # print("pass intersection", start, next)
                #     break
                start = next

            if i == ord - 1:
                # print("one iteration")
                iteration += 1
                if iteration >= 50:
                    break
                if usedout == 0:
                    i = -1
                    usedout = 1
                else:
                    break
            i += 1
        # return 0, self.adjustDir(new_lo, new_up, next), iteration
        adjustDir, inOrOut, scope = self.adjustDirNew(new_lo, new_up, next)
        # for i in range(self.A.shape[0]):
        #     if -0.0001 < adjustDir[i] < 0.0001:
        #         adjustDir[i] = 0
        #         result = 1
        return result, adjustDir, inOrOut, scope, iteration

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
            # if point[i] <= low[i] - 0.001:
            if point[i] <= low[i] - 0.001:
                result = 0
                return result
            # if point[i] >= up[i] + 0.001:
            if point[i] >= up[i] + 0.001:
                result = 0
                return result
        return result

    def checkEmpty(self, lo, up):
        for i in range(self.A.shape[0]):
            if lo[i] >= up[i]:
                return 1
        return 0

    def preCheck(self, D_lo, D_up, E_lo, E_up):
        for i in range(self.A.shape[0]):
            # empty set
            if D_lo[i] >= D_up[i]:
                return -1
            # impossible intersection
            if D_lo[i] >= E_up[i] or D_up[i] <= E_lo[i]:
                return 1
        return 0

    def cropBox(self, D_lo, D_up, E_lo, E_up):
        new_lo = []
        new_up = []
        for i in range(self.A.shape[0]):
            if E_lo[i] <= D_lo[i] and E_up[i] >= D_up[i]:
                new_up.append(D_up[i])
                new_lo.append(D_lo[i])
            elif E_lo[i] >= D_lo[i] and E_up[i] <= D_up[i]:
                new_up.append(E_up[i])
                new_lo.append(E_lo[i])
            elif E_lo[i] <= D_lo[i] and E_up[i] <= D_up[i]:
                new_up.append(E_up[i])
                new_lo.append(D_lo[i])
            elif E_lo[i] >= D_lo[i] and E_up[i] >= D_up[i]:
                new_up.append(D_up[i])
                new_lo.append(E_lo[i])

        new_lo = np.array(new_lo)
        new_up = np.array(new_up)
        return new_lo, new_up

    def newEqual(self, a, b):
        if -0.001 <= a - b <= 0.001:
            return 1
        else:
            return 0

    def adjustDir(self, D_lo, D_up, point):
        adjustDir = np.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            if point[i] <= D_lo[i]:
                adjustDir[i] = point[i] - D_lo[i]
            elif point[i] >= D_up[i]:
                adjustDir[i] = point[i] - D_up[i]
        return adjustDir

    # inOrDe 0: decrease, 1: increase
    def adjustTau(self, pOrN, start, end):
        self.numAdjust += 1
        startTime = time.time()
        # box's center movements
        deltaCs = []
        # box's edge movements
        deltaEs = []
        # corresponding steps
        for d in range(self.max_step):
            deltaC = []
            deltaE = []
            # corresponding support vectors
            for j in range(self.A.shape[0]):
                t = np.zeros(self.A.shape)
                for i in range(len(pOrN)):
                    t += self.A_i[i] * pOrN[-i]
                t = 0.5 * self.A_i[d] @ t
                t = t.T @ self.l[j]
                deltaC.append(t.T)

                # select one column in 1/2 A_i
                b = (0.5 * self.A_i[d].T)[:, j]
                # corresponding generators
                for i in range(self.A.shape[0]):
                    b[i] = b[i] * self.sNorm[d][j][i]
                sum_A_i = np.zeros(self.A.shape)
                for i in range(len(pOrN)):
                    sum_A_i += self.A_i[i]
                # select one row in sum_A_i
                a = sum_A_i[j]
                deltaE.append(np.sum(b) * a)
            deltaCs.append(deltaC)
            deltaEs.append(deltaE)

        self.deltaCs = deltaCs
        self.deltaEs = deltaEs

        if start != 0:
            objStep = start - 1
            deltaTau = self.getDeltaTau(objStep)
        else:
            objStep = self.max_step - 1
            for i in range(self.max_step):
                if np.any(self.adjustDirs[i]):
                    objStep = i
            # deltaTau = self.getDeltaTau(objStep)
            deltaTau = self.getDeltaTau(objStep)
        endTime = time.time()
        self.timeAdjust += endTime - startTime
        print("avg adjust time: ", self.timeAdjust / self.numAdjust)
        return deltaTau

    def adjustTauNew(self, pOrN, start, end, inOrDe, detector):
        self.detector = detector
        self.numAdjust += 1
        startTime = time.time()
        # box's center movements
        deltaCs = []
        # box's edge movements
        deltaEs = []
        # corresponding steps
        for d in range(self.max_step):
            deltaC = []
            deltaE = []
            # corresponding support vectors
            for j in range(self.A.shape[0]):
                t = np.zeros(self.A.shape)
                for i in range(len(pOrN)):
                    t += self.A_i[i] * pOrN[-i]
                t = 0.5 * self.A_i[d] @ t
                t = t.T @ self.l[j]
                deltaC.append(t.T)

                # select one column in 1/2 A_i
                b = (0.5 * self.A_i[d].T)[:, j]
                # corresponding generators
                for i in range(self.A.shape[0]):
                    b[i] = b[i] * self.sNorm[d][j][i]
                sum_A_i = np.zeros(self.A.shape)
                for i in range(len(pOrN)):
                    sum_A_i += self.A_i[i]
                # select one row in sum_A_i
                a = sum_A_i[j]
                deltaE.append(np.sum(b) * a)
            deltaCs.append(deltaC)
            deltaEs.append(deltaE)

        self.deltaCs = deltaCs
        self.deltaEs = deltaEs

        if inOrDe == 0:
            # increase k
            if start != 0:
                objStep = start - 1
                deltaTau = self.getDeltaTauIncreaseK(objStep)
            else:
                objStep = 0
                # objStep = self.max_step - 1
                exist = 0
                for i in range(self.max_step):
                    # Probable intersection
                    if np.any(self.inOrOuts[i]):
                        objStep = i
                        exist = 1
                        # break
                if exist == 0:
                    for i in range(self.max_step):
                        if np.any(self.adjustDirs[i]) and self.emptySet[i] == 0:
                            objStep = i
                    if objStep == 0:
                        for i in range(self.max_step):
                            if i <= objStep:
                                continue
                            # empty set
                            if np.any(self.adjustDirs[i]) and self.emptySet[i] == 1:
                                objStep = i
                                break
                    # # Impossible to intersect
                    # if objStep == 0:
                    #     for i in range(self.max_step):
                    #         if np.any(self.adjustDirs[i]) and self.emptySet[i] == 0:
                    #             objStep = i

                # deltaTau = self.getDeltaTau(objStep)
                deltaTau = self.getDeltaTauIncreaseK(objStep)
        else:
            # decrease k
            objStep = start + 1
            # objStep = end - 6
            deltaTau = self.getDeltaTauDecreaseKAllDim(objStep)

        endTime = time.time()
        self.timeAdjust += endTime - startTime
        print("avg adjust time: ", self.timeAdjust / self.numAdjust)
        return deltaTau


    def getSNorm(self, D: Zonotope):
        sNorms = []
        # corresponding support vectors
        for i in range(self.A.shape[0]):
            sNorm = []
            t = D.g.T @ self.l[i]
            # corresponding generators
            for j in range(self.A.shape[0]):
                if t[j] >= 0:
                    sNorm.append(1)
                else:
                    sNorm.append(-1)
            sNorms.append(sNorm)
        return sNorms

    def getDeltaTau(self, d):
        deltaTau = np.zeros(self.A.shape[0])
        adjustDim = np.zeros(self.A.shape[0])

        for i in range(self.A.shape[0]):
            if self.adjustDirs[d][i] != 0:
                adjustDim[i] = 1
        numDim = int(np.sum(adjustDim))

        # number of coefficient = number of dimension which need to be adjusted
        coefficient = np.zeros([numDim, numDim])
        newAdjustDir = np.zeros(numDim)
        k = 0
        for i in range(self.A.shape[0]):
            # delta C + delta E
            if self.adjustDirs[d][i] > 0:
                newAdjustDir[k] = self.adjustDirs[d][i]
                for j in range(numDim):
                    # coefficient[k][j] = self.deltaCs[d][k][j] - self.deltaEs[d][k][j]
                    coefficient[k][j] = self.deltaCs[d][i][j] - self.deltaEs[d][i][j]
            # delta C - delta E
            elif self.adjustDirs[d][i] < 0:
                newAdjustDir[k] = self.adjustDirs[d][i]
                for j in range(numDim):
                    # coefficient[k][j] = self.deltaCs[d][k][j] + self.deltaEs[d][k][j]
                    coefficient[k][j] = self.deltaCs[d][i][j] + self.deltaEs[d][i][j]
        result = np.linalg.pinv(coefficient) @ newAdjustDir
        k = 0
        for i in range(self.A.shape[0]):
            if adjustDim[i] == 0:
                deltaTau[i] = 0
            else:
                deltaTau[i] = result[k]
                k += 1

        return deltaTau

    # get delta tau for increasing recoveryability k
    def getDeltaTauIncreaseK(self, d):
        deltaTau = np.zeros(self.A.shape[0])
        print("inOrOuts", self.inOrOuts[d])
        print("self.adjustDirs[d]", self.adjustDirs[d])
        print("empty set", self.emptySet[d])
        if not np.any(self.inOrOuts[d]):
            print("no in")
        if np.sum(self.inOrOuts[d]) != 0:
            numDim = self.A.shape[0] - (np.sum(self.inOrOuts[d]))
            numDim = int(numDim)
            adjustDim = []

            for i in range(self.A.shape[0]):
                if self.inOrOuts[d][i] == 0:
                    adjustDim.append(i)
            adjustDim = np.array(adjustDim)
        else:
            numDim = 0
            adjustDim = []
            for i in range(self.A.shape[0]):
                if self.adjustDirs[d][i] != 0:
                    numDim += 1
                    adjustDim.append(i)
            adjustDim = np.array(adjustDim)

        # number of coefficient = number of dimension which need to be adjusted
        coefficient = np.zeros([numDim, numDim])
        newAdjustDir = np.zeros(numDim)
        k = 0
        for i in range(self.A.shape[0]):
            # delta C + delta E
            if self.adjustDirs[d][i] > 0 and self.inOrOuts[d][i] == 0:
                newAdjustDir[k] = self.adjustDirs[d][i]
                for j in range(numDim):
                    # coefficient[k][j] = self.deltaCs[d][k][j] - self.deltaEs[d][k][j]
                    coefficient[k][j] = self.deltaCs[d][i][adjustDim[j]] + self.deltaEs[d][i][adjustDim[j]]
                k += 1
            # delta C - delta E
            elif self.adjustDirs[d][i] < 0 and self.inOrOuts[d][i] == 0:
                newAdjustDir[k] = self.adjustDirs[d][i]
                for j in range(numDim):
                    # coefficient[k][j] = self.deltaCs[d][k][j] + self.deltaEs[d][k][j]
                    coefficient[k][j] = self.deltaCs[d][i][adjustDim[j]] - self.deltaEs[d][i][adjustDim[j]]
                k += 1
        result = np.linalg.pinv(coefficient) @ newAdjustDir
        for i in range(np.shape(result)[0]):
            deltaTau[adjustDim[i]] = result[i]
        return deltaTau

    def getDeltaTauIncreaseKNew(self, d):
        deltaTau = np.zeros(self.A.shape[0])
        print("inOrOuts", self.inOrOuts[d])
        print("self.adjustDirs[d]", self.adjustDirs[d])
        print("empty set", self.emptySet[d])
        if not np.any(self.inOrOuts[d]):
            print("no in")
        if np.sum(self.inOrOuts[d]) != 0:
            numDim = self.A.shape[0] - (np.sum(self.inOrOuts[d]))
            numDim = int(numDim)
            adjustDim = []

            for i in range(self.A.shape[0]):
                if self.inOrOuts[d][i] == 0:
                    adjustDim.append(i)
            adjustDim = np.array(adjustDim)
        else:
            numDim = 0
            adjustDim = []
            for i in range(self.A.shape[0]):
                if self.adjustDirs[d][i] != 0:
                    numDim += 1
                    adjustDim.append(i)
            adjustDim = np.array(adjustDim)

    # get delta tau for decreasing recoveryability k
    # only adjust one dimension
    def getDeltaTauDecreaseK(self, d):
        print("self.adjustDirs[d]", self.adjustDirs[d])
        deltaTau = np.zeros(self.A.shape[0])
        supDim = np.argmin(self.scopes[d])
        argminTau = self.detector.minTau()

        coefficient = np.zeros(self.A.shape[0])
        if self.adjustDirs[d][supDim] > 0:
            coefficient = np.array(self.deltaCs[d][supDim]) - np.array(self.deltaEs[d][supDim])
        elif self.adjustDirs[d][supDim] < 0:
            coefficient = np.array(self.deltaCs[d][supDim]) + np.array(self.deltaEs[d][supDim])
        t = coefficient
        for i in range(self.A.shape[0]):
            t[i] = np.abs(t[i])
        adjustDim = np.argmax(t)
        if adjustDim == argminTau and self.detector.tao[argminTau] <= 0.032/2:
            t[adjustDim] = 0
            adjustDim = np.argmax(t)
        deltaTau[adjustDim] = self.adjustDirs[d][supDim] / coefficient[adjustDim]
        return deltaTau

    def getDeltaTauDecreaseKAllDim(self, d):
        print("self.adjustDirs[d]", self.adjustDirs[d])
        deltaTau = np.zeros(self.A.shape[0])
        f = self.adjustDirs[d]
        for i in range(self.A.shape[0]):
            f[i] = np.abs(self.adjustDirs[d][i])
        # supDim = np.argmin(self.scopes[d])
        supDim = np.argmin(f)
        print("self.adjustDirs[d][supDim]", self.adjustDirs[d][supDim])
        print("supdim", supDim)
        coefficients = np.zeros(self.A.shape[0])
        if self.adjustDirs[d][supDim] > 0:
            coefficients = np.array(self.deltaCs[d][supDim]) - np.array(self.deltaEs[d][supDim])
        elif self.adjustDirs[d][supDim] < 0:
            coefficients = np.array(self.deltaCs[d][supDim]) + np.array(self.deltaEs[d][supDim])
        coefficient = np.sum(coefficients)
        print("coefficients", coefficients)
        adjustDims = []
        coefficient = 0
        if self.adjustDirs[d][supDim] > 0:
            adjustDim = np.argmax(coefficients)
            # for i in range(self.A.shape[0]):
            #     if coefficients[i] > 0:
            #         adjustDims.append(i)
            #         coefficient += coefficients[i]
        else:
            adjustDim = np.argmin(coefficients)
            # for i in range(self.A.shape[0]):
            #     if coefficients[i] < 0:
            #         adjustDims.append(i)
            #         coefficient += coefficients[i]
        delta = self.adjustDirs[d][supDim] / coefficients[adjustDim]
        # delta = self.adjustDirs[d][supDim] / coefficient
        deltaTau[adjustDim] = delta
        # for i in range(len(adjustDims)):
        #     deltaTau[i] = delta
        # tau + deltaTau
        return deltaTau

    # probably intersection
    def adjustDirNew(self, D_lo, D_up, point):
        adjustDir = np.zeros(self.A.shape[0])
        inOrOut = np.zeros(self.A.shape[0])
        scope = np.zeros(self.A.shape[0])
        for i in range(self.A.shape[0]):
            if point[i] <= D_lo[i]:
                adjustDir[i] = point[i] - D_lo[i]
                inOrOut[i] = 0
            elif point[i] >= D_up[i]:
                adjustDir[i] = point[i] - D_up[i]
                inOrOut[i] = 0
            elif D_lo[i] < point[i] < D_up[i]:
                inOrOut[i] = 1
                if D_up[i] - point[i] > point[i] - D_lo[i]:
                    adjustDir[i] = point[i] - D_lo[i]
                else:
                    adjustDir[i] = point[i] - D_up[i]
            scope[i] = np.abs(adjustDir[i]) / (D_up[i] - D_lo[i])
        return adjustDir, inOrOut, scope


    def adjustDirsNN(self, D_lo, D_up, E_lo, E_up):
        adjustDir = np.zeros(self.A.shape[0])
        inOrOut = np.zeros(self.A.shape[0])
        scope = np.zeros(self.A.shape[0])

        for i in range(self.A.shape[0]):
            if D_lo[i] >= E_up[i]:
                adjustDir[i] = E_up[i] - D_lo[i]
            elif D_up[i] <= E_lo[i]:
                adjustDir[i] = E_lo[i] - D_up[i]
            if -0.001 < adjustDir[i] < 0.001:
                adjustDir[i] = 0
            scope[i] = adjustDir[i] / (D_up[i] - D_lo[i])

        return adjustDir, inOrOut, scope

    def emtpySetDir(self, D_lo, D_up, E_lo, E_up):
        adjustDir = np.zeros(self.A.shape[0])

        for i in range(self.A.shape[0]):
            if D_lo[i] > D_up[i]:
                adjustDir[i] = D_up[i] - D_lo[i]
        return adjustDir