import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
import time
import threading

def reach_E(A_i_B_U):
    E = []
    for i in range(2):
        if i == 0:
            E.append(A_i_B_U[0])
        else:
            E.append(E[-1] + A_i_B_U[i])
    return E

A = np.array([[ 0.99961363, 0.17447524, -0.64387563, -0.01430613],[-0.00000503,  0.9799524,   0.00000162,  0.0178227 ],
              [-0.,          0.00016173,  1.,          0.01978653],[-0.00000004,  0.01605999,  0.00000001,  0.97877772]])
B = np.array([[ 0.00341851], [-0.00007488], [-0.00003575], [-0.00356192]])

U = Zonotope.from_box(np.ones(1) * -10, np.ones(1) * 10)
A_i = [np.eye(A.shape[0])]
for i in range(20):
    A_i.append(A @ A_i[-1])
A_i_B_U = [val @ B @ U for val in A_i]
E = []
for i in range(20):
    if i == 0:
        E.append(A_i_B_U[0])
    else:
        E.append(E[-1] + A_i_B_U[i])



numFinish = 0
lock = threading.Lock()
event = threading.Event()
start = threading.Event()
result = np.zeros(20)

def check(lk, step:int):
    global A_i_B_U
    global numFinish
    first = Zonotope.from_box(np.ones(4) * -0.02, np.ones(4) * 0.02)
    j = cp.Parameter(4, value=A_i_B_U[step].c)
    h = cp.Parameter((4, 1), value=A_i_B_U[step].g)
    alpha = np.ones([h.shape[0]])
    beta = cp.Variable([h.shape[1]], name="beta")
    constraints = [
        (beta[k] <= 1) for k in range(h.shape[1])
    ]
    constraints += [
        (beta[k] >= -1) for k in range(h.shape[1])
    ]
    constraints += [
        j + h @ beta <= first.c + first.g @ alpha
    ]
    constraints += [
        j + h @ beta >= -1 * (first.c + first.g @ alpha)
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    print(step)

    while True:
        if start.is_set():
            event.clear()
            A_i_B_U[step].c = A_i_B_U[step].c * 1.1
            A_i_B_U[step].g = A_i_B_U[step].g * 1.1
            j.value = A_i_B_U[step].c
            h.value = A_i_B_U[step].g
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
        checker = threading.Thread(target=check, args=(lock, i,))
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
            event.wait()
