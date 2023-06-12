import os.path
import numpy as np
import cvxpy as cp
from simulators.linear.platoon import Platoon
class Authenticate:
    def __int__(self, exp, n, p=0.002, v=0.002, inf=0.5):
        self.u = None
        self.states = None
        self.feedbacks = None
        self.INF = inf
        self.m = exp.sysd.A.shape[0]
        self.n = n
        self.timestep = 2 * self.m
        self.A = exp.sysd.A
        self.B = exp.sysd.B
        self.delta = self.getDelta(p, v)
        self.A_k = self.getA_k()

    def getDelta(self, p, v):
        d = np.zeros([self.timestep, self.m])
        for i in range(self.timestep):
            if i == 0:
                d[0] = np.zeros(self.m)
            else:
                if i == 1:
                    d[1] = np.ones(self.m) * 0.002
                else:
                    d[i] = abs(self.A) @ d[i - 1] + d[1]
        for i in range(self.m):
            d[i] += np.ones(self.m) * 0.002
        return d

    def getA_k(self):
        a_k = np.zeros([self.timestep, self.m, self.m])
        for i in range(self.timestep):
            if i == 0:
                a_k[0] = np.eye(self.m)
            else:
                a_k[i] = self.A @ a_k[i - 1]
        return a_k

    def getInputs(self, inpu):
        self.inputs = inpu

    def getStates(self, stat):
        self.states = stat

    def getFeedbacks(self, feed):
        self.feedbacks = feed

    def getAuth(self):
        auth = 0
        return auth

    def getBound(self):
        bound = 0
        return bound