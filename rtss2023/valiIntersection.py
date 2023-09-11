import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
from utils.Baseline import Platoon
import time

def checkin(low, up, point):
    result = 1
    for i in range(low.shape[0]):
        if point[i] <= low[i]:
            result = 0
            return result
        if point[i] >= up[i]:
            result = 0
            return result
    return result

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
        # E.append(A_i_B_U[0])

# temp = E[4]
temp = E[4].order_reduction(7)
temp.g = temp.g * 2

D_low = [-0.6327720639435364, -0.602399556735256, -0.6596764214473849, -1.3641168110318211, -1.2465597978222978, -1.4271499822702531, -1.3953298454509189]
D_up = [-0.03429068321966455, 0.03286102079409425, -0.05438234810840625, -0.3470178501868151, -0.2749886047735577, -0.46296348863231485, -0.40030780345735284]

D_low = np.array(D_low)
D_up = np.array(D_up)

steps = (D_up - D_low) / 2

dim = temp.g.shape[1]
# temp = E[9]

beta = cp.Variable([dim], name="beta")

constraints = [
    (beta[k] <= 1) for k in range(dim)
]
constraints += [
    (beta[k] >= -1) for k in range(dim)
]
constraints += [
    temp.c + temp.g @ beta <= D_up
]
constraints += [
    temp.c + temp.g @ beta >= D_low
]
problem = cp.Problem(cp.Minimize(0), constraints)
result = problem.solve()
print(beta.value)
# print("result point")
# print(temp.g @ beta.value)

E_inv = np.linalg.inv(temp.g)
# print(E_inv @ temp.g @ beta.value)
t = E_inv @ temp.g
print("up")
print(E_inv @ D_up)
print("low"),
print(E_inv @ D_low)
D_1 = D_up
D_1[1] = 0
print(E_inv @ D_1)
D_2 = D_up
D_2[2] = D_low[2]
D_2[3] = D_low[3]
print(E_inv @ D_2)

D_3 = (D_up + D_low)/2
print(E_inv @ D_3)


# print(temp.g @ E_inv @ D_up)
# print(temp.g @ E_inv @ D_low)

# start = (D_low + D_up)/2
# alp = E_inv @ start
# delta = np.zeros(7)
# for i in range(7):
#     if alp[i] >= 1:
#         delta[i] = alp[i] - 1
#     elif alp[i] <= -1:
#         delta[i] = alp[i] + 1
# dir = temp.g @ delta
# dir = -dir
# dirNorm = abs(dir)
# num = np.argsort(dirNorm)[-1]
# if start[num] + dir[num] >= D_up[num]:
#     step = D_up[num] - start[num]
# elif start[num] + dir[num] <= D_low[num]:
#     step = D_low[num] - start[num]
# else:
#     step = dir[num]
# flags = 0
# while(True):
#     start[num] = start[num] + step
#     print("start", start)
#     alp = E_inv @ start
#     print("alp", alp)
#     delta = np.zeros(7)
#     end = 1
#     for i in range(7):
#         if alp[i] >= 1:
#             delta[i] = alp[i] - 1
#             end = 0
#         elif alp[i] <= -1:
#             delta[i] = alp[i] + 1
#             end = 0
#     if end == 1:
#         print("intersect")
#         break
#     dir = temp.g @ delta
#     dir = -dir
#     dirNorm = abs(dir)
#     for i in range(7):
#         num = np.argsort(dirNorm)[-1 - i]
#         if start[num] + dir[num] >= D_up[num]:
#             step = D_up[num] - start[num]
#         elif start[num] + dir[num] <= D_low[num]:
#             step = D_low[num] - start[num]
#         else:
#             step = dir[num]
#         if step <= -1e-15 or step >= 1e-15:
#             print("step, num", step, num)
#             break
#         if i == 6 and (step >= -1e-15 or step <= 1e-15):
#             flags += 1
#             if flags == 7:
#                 print("no intersect")
#                 exit()
#             if alp[flags] >= 1:
#                 delta[flags] = alp[flags] + 1
#             elif alp[flags] <= -1:
#                 delta[flags] = alp[flags] - 1
#             i = 0
#             # start = (D_low + D_up)/2
#             # start[flags] = ((D_low + D_up)/2)[flags] + steps[flags]
#             # step = 0
#             # flags += 1


# t = np.ones(dim).reshape(-1, 1).T
# G = np.append(temp.g, np.eye(dim), axis=0)
# lo = np.append(D_low, -np.ones(dim))
# up = np.append(D_up, np.ones(dim))
# E_inv = np.linalg.pinv(G)
# k = E_inv @ G
# print(k)
# print("lo")
# lo = lo.reshape(-1, 1)
# up = up.reshape(-1, 1)
# print(E_inv @ lo)
# print("up")
# print(E_inv @ up)


# dirct = (D_up + D_low) / 2
# print(np.linalg.norm(dirct))
# generators = np.zeros(dim)
# pOrN = np.zeros(dim)
# print("generators")
# for i in range(dim):
#     t = np.dot(temp.g[:, i].T, dirct)
#     generators[i] = abs(t)
#     print(generators)
#     if t < 0:
#         pOrN[i] = -1
#     else:
#         pOrN[i] = 1
# gSort = np.argsort(generators)
# gSort = gSort[::-1]
# print("gsort", gSort)
# t = np.zeros(dim)
# for i in range(dim):
#     flag = gSort[i]
#     t += temp.g[:, flag].T * pOrN[flag]
#     print(np.linalg.norm(dirct - t))
#     end = checkin(D_low, D_up, t)
#     if end:
#         print("intersect")
# print("not intersect")