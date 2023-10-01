import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
from utils.Baseline import Platoon
import time

np.set_printoptions(suppress=True)

def getT(a, b):
    # b is generator
    k = np.dot(a, b)/np.dot(b, b)
    return k

def closestPoint(low, up, point):
    re = np.zeros(low.shape[0])
    for i in range(low.shape[0]):
        if low[i] <= point[i] <= up[i]:
            re[i] = point[i]
        elif point[i] < low[i]:
            re[i] = low[i]
        elif point[i] > up[i]:
            re[i] = up[i]
    return re

def checkinBox(low, up, point):
    result = 1
    for i in range(low.shape[0]):
        if point[i] <= low[i] - 0.01:
            result = 0
            return result
        if point[i] >= up[i] + 0.01:
            result = 0
            return result
    return result

def checkPass(a, b, boxC, boxG):
    # a, b point position

    # transfer from Zonotope to Box
    a = a - boxC
    a = np.divide(a, boxG)
    b = b - boxC
    b = np.divide(b, boxG)

    up = np.ones(7)
    up = up - b
    low = -np.ones(7) - b

    for i in range(7):
        t = a[i] - b[i]
        if t > 0:
            up[i] = up[i] / t
            low[i] = low[i] / t
        elif t < 0:
            temp = up[i]
            up[i] = low[i] / t
            low[i] = temp / t

    f_up = 0
    f_low = 0
    signal = 1
    for i in range(7):
        if i == 0:
            f_up = up[i]
            f_low = low[i]
        else:
            if f_up >= up[i] and f_low <= low[i]:
                f_up = up[i]
                f_low = low[i]
            elif f_up <= up[i] and f_low <= low[i]:
                f_low = low[i]
            elif f_up >= up[i] and f_low >= low[i]:
                f_up = up[i]
            elif f_low >= up[i] or f_up <= low[i]:
                signal = -1
                break
    if f_low > 1 or f_up < 0:
        signal = -1
    return f_low, f_up, signal


A = Platoon.model.sysd.A
B = Platoon.model.sysd.B

U = Zonotope.from_box(np.ones(4) * -10, np.ones(4) * 10)
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

# Zonotope E for test
temp = E[8]
ord = temp.g.shape[1]
dim = 7

# E_bound
l = np.eye(A.shape[0])
E_up = []
E_low = []
for j in range(7):
    E_low.append(temp.support(l[j], -1))
    E_up.append(temp.support(l[j]))
print("E_low")
print(E_low)
print("E_up")
print(E_up)
# 8
D_low = [-0.5127183015494863, -0.6167577998252325, -0.4805234631894035, -1.4858918910093102, -1.295625944941733, -1.3699859783507793, -1.2395331015932851]
D_up = [-0.07516555595502172, -0.17636215476439565, -0.03138843058711416, -0.47209028044676304, -0.31121943560775733, -0.40006561696619336, -0.27035857692944454]
# 4
# D_low = [-0.6327720639435364, -0.602399556735256, -0.6596764214473849, -1.3641168110318211, -1.2465597978222978, -1.4271499822702531, -1.3953298454509189]
# D_up = [-0.03429068321966455, 0.03286102079409425, -0.05438234810840625, -0.3470178501868151, -0.2749886047735577, -0.46296348863231485, -0.40030780345735284]
D_low = np.array(D_low)
D_up = np.array(D_up)

# # Crop the box
# new_low = []
# new_up = []
# for i in range(dim):
#     # contain
#     if E_low[i] <= D_low[i] and E_up[i] >= D_up[i]:
#         new_up.append(D_up[i])
#         new_low.append(D_low[i])
#         # select.append(0)
#     elif E_low[i] >= D_low[i] and E_up[i] <= D_up[i]:
#         new_up.append(E_up[i])
#         new_low.append(E_low[i])
#         # select.append(0)
#     elif E_low[i] <= D_low[i] and E_up[i] <= D_up[i]:
#         new_up.append(E_up[i])
#         new_low.append(D_low[i])
#         # select.append()
#     elif E_low[i] >= D_low[i] and E_up[i] >= D_up[i]:
#         new_up.append(D_up[i])
#         new_low.append(E_low[i])
#     elif E_up[i] <= D_low[i] or E_low[i] >= D_up[i]:
#         print("no intersection")
#         exit(1)
# new_up = np.array(new_up)
# new_low = np.array(new_low)
# print("new_up")
# print(new_up)
# print("new_low")
# print(new_low)

new_low = D_low
new_up = D_up


box = Zonotope.from_box(new_low, new_up)
boxG = new_up - box.c

# LP
beta = cp.Variable([ord], name="beta")

constraints = [
    (beta[k] <= 1) for k in range(ord)
]
constraints += [
    (beta[k] >= -1) for k in range(ord)
]
constraints += [
    ((temp.c + temp.g @ beta)[k] <= D_up[k]) for k in range(dim)
]
constraints += [
    ((temp.c + temp.g @ beta)[k] >= D_low[k]) for k in range(dim)
]
problem = cp.Problem(cp.Minimize(0), constraints)
result = problem.solve()
print("beta.value")
print(beta.value)
# print("result point")
# result_point = temp.g @ beta.value
# print(result_point)

start = temp.c
dir = np.zeros(ord)
used = np.zeros(ord)
move = np.zeros(ord)
usedout = 1
i = 0
iteration = 0
starttime = time.time()

# firstLoop = np.zeros(ord)
# while i < ord:
#     t = getT(box.c - start, temp.g[:, i])
#     i += 1

while i < ord:
    # t = getT(closestPoint(new_low, new_up, start) - start, temp.g[:, i])
    t = getT(box.c - start, temp.g[:, i])
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
        next = start + move[i] * temp.g[:, i]
        # print(next)
        # distance = np.linalg.norm(next - closestPoint(new_low, new_up, next))
        # print("distance", distance)
        re = checkinBox(new_low, new_up, next)
        if re == 1:
            # print("intersect point", next)
            break
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
endtime = time.time()
print("time", endtime-starttime)
print("iteration", iteration)
print("finish explore")
print(dir)

