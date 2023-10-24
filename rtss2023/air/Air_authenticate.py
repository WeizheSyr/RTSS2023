import os.path
import numpy as np
import cvxpy as cp
from simulators.linear.platoon import Platoon


class Authenticate:
    def __init__(self, exp, n, p=0.002, v=0.002, inf=100):
        self.u = None
        self.y = None
        self.INF = inf
        self.m = exp.sysd.A.shape[0]    # dimension of A
        self.n = n                      # dimension of u
        self.timestep = self.m      # dimension of timestep

        # optimal value
        self.x = np.zeros(self.m)
        self.gama = np.zeros(self.m)
        self.E = np.zeros([self.timestep, self.m])
        self.bound = np.zeros([self.m, 2])

        self.A = exp.sysd.A
        self.B = exp.sysd.B
        self.delta = self.getDelta(p, v)
        self.A_k = self.getA_k()
        self.U = None

    def getDelta(self, p, v):
        d = np.zeros([self.timestep, self.m])
        for i in range(self.timestep):
            if i == 0:
                d[0] = np.zeros(self.m)
            else:
                if i == 1:
                    d[1] = np.ones(self.m) * p
                else:
                    d[i] = abs(self.A) @ d[i - 1] + d[1]
        return d

    def getA_k(self):
        a_k = np.zeros([self.timestep, self.m, self.m])
        for i in range(self.timestep):
            if i == 0:
                a_k[0] = np.eye(self.m)
            else:
                a_k[i] = self.A @ a_k[i - 1]
        return a_k

    def getU(self):
        U = np.zeros([self.timestep, self.m])
        for i in range(self.timestep):
            if i == 0:
                U[0] = np.zeros([self.m])
            else:
                U[i] = self.B @ self.u[i - 1] + self.A @ U[i - 1]
        return U

    def getInputs(self, inpu):
        self.u = inpu

    def getFeedbacks(self, feed):
        self.y = feed

    def getAuth(self, att=0):
        self.U = self.getU()

        x = cp.Variable([self.m], name="x")
        gama = cp.Variable([self.m], name="gama", boolean=True)
        E = cp.Variable([self.timestep, self.m], name="E")

        obj = cp.sum(gama)
        constraints = [
            (gama[k] <= 0) for k in range(self.m)
        ]

        # under attack dimension
        del constraints[att]
        constraints += [
            gama[att] <= 1
        ]
        constraints += [
            (self.y[k] - self.U[k] - (self.A_k[k] @ x) - E[k] <= self.delta[k]) for k in range(self.timestep)
        ]
        constraints += [
            (self.y[k] - self.U[k] - (self.A_k[k] @ x) - E[k] >= -self.delta[k]) for k in range(self.timestep)
        ]
        constraints += [
            (E[:, k] <= self.INF * gama[k] * np.ones(self.timestep)) for k in range(self.m)
        ]
        constraints += [
            (E[:, k] >= -1 * self.INF * gama[k] * np.ones(self.timestep)) for k in range(self.m)
        ]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.SCIPY)

        # print('x.value', x.value)
        for i in range(self.m):
            self.x[i] = x.value[i]
        for i in range(self.m):
            self.gama[i] = gama.value[i]
        for i in range(self.timestep):
            for j in range(self.m):
                self.E[i][j] = E.value[i][j]

    def getBound(self, dim=0, att=0):
        x1 = cp.Variable([self.m], name="x1")
        E1 = cp.Variable([self.timestep, self.m], name="E1")
        beta = cp.Variable([self.m], name="beta", boolean=True)

        obj1 = x1[dim] - self.x[dim]

        temp = cp.Variable([self.timestep, self.m])

        temp1 = np.zeros([self.timestep, self.m])
        for i in range(self.timestep):
            temp1[i] = self.A_k[i] @ self.x

        constraints1 = [
            cp.sum(beta) <= 1
        ]
        constraints1 += [
            (self.y[k] - self.U[k] - (self.A_k[k] @ x1) - E1[k] <= self.delta[k]) for k in range(self.timestep)
        ]
        constraints1 += [
            (self.y[k] - self.U[k] - (self.A_k[k] @ x1) - E1[k] >= -self.delta[k]) for k in range(self.timestep)
        ]
        constraints1 += [
            (E1[:, k] <= self.INF * beta[k] * np.ones(self.timestep)) for k in range(self.m)
        ]
        constraints1 += [
            (E1[:, k] >= -1 * self.INF * beta[k] * np.ones(self.timestep)) for k in range(self.m)
        ]

        upB = cp.Problem(cp.Maximize(obj1), constraints1)
        lowB = cp.Problem(cp.Minimize(obj1), constraints1)
        upB.solve()
        lowB.solve()

        self.bound[dim, 0] = lowB.value
        self.bound[dim, 1] = upB.value

    def getAllBound(self, att=0):
        for i in range(self.m):
            self.getBound(i, att)