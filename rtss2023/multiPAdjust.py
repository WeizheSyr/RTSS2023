import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
import time
import threading
import multiprocessing as mp
from utils.Baseline import Platoon

def reach_E(A_i_B_U):
    E = []
    for i in range(2):
        if i == 0:
            E.append(A_i_B_U[0])
        else:
            E.append(E[-1] + A_i_B_U[i])
    return E

A = Platoon.model.sysd.A
B = Platoon.model.sysd.B

U = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)
A_i = [np.eye(A.shape[0])]
for i in range(20):
    A_i.append(A @ A_i[-1])
A_i_B_U = [val @ B @ U for val in A_i]
E = []
for i in range(20):
    if i == 0:
        E.append(A_i_B_U[0])
    else:
        # E.append(E[-1] + A_i_B_U[i])
        E.append(A_i_B_U[0])
d = 10
# authTimestep + self.auth.timestep
k = 9

# A^d
A_d = [np.eye(A.shape[0])]
for i in range(20):
    A_d.append(A @ A_d[-1])

# I + A^(k-1)
A_k_1 = []
for i in range(k):
    if i == 0:
        A_k_1.append(np.eye(A.shape[0]))
    else:
        A_k_1.append(A_k_1[-1] + A_d[i - 1])

# F()
F = A_d[d - 1] @ A_k_1[k - 1]

D_low = [-0.22060118825057262, -0.2201856099204006, -0.028929731712184048, -1.1590693513020622, -0.8727438691416309, -0.7056735968203992, -0.43328699395346926]
D_up = [-0.4023044399373612, -0.3933730907986046, -0.19501647114507392, -1.316595325577472, -1.0624287153104497, -0.8946908257562018, -0.5895528329640749]

numFinish = 0
lock = mp.Lock()
event = mp.Event()
start = mp.Event()
result = np.zeros(20)

def check(lk, step:int, start, event):
    global A_i_B_U
    global numFinish

    tau = cp.Variable(7, name="delta_tau")
    alpha = cp.Variable([E[step - 1].g.shape[1]], name="alpha")

    tempc = cp.Parameter(7, value=E[step - 1].c)
    tempg = cp.Parameter((7,4), value=E[step - 1].g)

    constraints = [
        ((tempc + tempg @ alpha)[f] <= D_up[f] + (F @ tau)[f] / 2) for f in range(A.shape[0])
    ]
    constraints += [
        ((tempc + tempg @ alpha)[f] >= D_low[f] - (F @ tau)[f] / 2) for f in range(A.shape[0])
    ]
    constraints += [
        (alpha[f] <= 1) for f in range(4)
    ]
    constraints += [
        (alpha[f] >= -1) for f in range(4)
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    print(step)

    while True:
        if start.is_set():
            event.clear()
            E[step - 1].c = E[step - 1].c * 1.1
            E[step - 1].g = E[step - 1].g * 1.1
            tempc = E[step - 1].c
            tempg = E[step - 1].g
            result[step] = problem.solve(warm_start=True)
            lk.acquire()
            numFinish += 1
            if numFinish >= 15:
                start.clear()
                event.set()
                numFinish = 0
            lk.release()
        else:
            start.wait()


if __name__ == '__main__':
    for i in range(15):
        print("checker: ", i)
        checker = mp.Process(target=check, args=(lock, i, start, event))
        checker.start()
    start_time = time.time()
    start.set()
    while True:
        if event.is_set():
            end_time = time.time()
            print("cost: ", end_time - start_time)
            start_time = time.time()
            start.set()
            event.clear()
        else:
            # print("main is waiting")
            event.wait()
