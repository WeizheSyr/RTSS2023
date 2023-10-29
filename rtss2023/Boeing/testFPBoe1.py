from utils.Baseline import Boeing
from utils.detector.windowBased import window
from FPEvaluBoe1 import FPEvaluation1
import numpy as np


FP_our = 0
FP_fixed = 0
largerThanK = 0
totalLength = 0
ourLength = 0

for i in range(30):
    # rseed = np.uint32(int(time.time()))
    # print(rseed)
    # np.random.seed(rseed)
    print("iteration num: ", i)
    tao = np.ones(5) * 0.1
    detector = window(tao, 5, 10)
    fixed_tao = np.ones(5) * 0.07
    fixed_detector = window(fixed_tao, 5, 10)
    exp = Boeing
    attack = np.zeros(50)
    attack_duration = 500

    sys = FPEvaluation1(detector=detector, fixed_detector=fixed_detector, exp=exp, attack=attack, attack_duration=attack_duration)
    print(sys.i)
    totalLength = totalLength + len(sys.fixed_klevels)
    ourLength = ourLength + len(sys.klevels)

    for j in range(len(sys.fixed_klevels)):
        if sys.fixed_klevels[j] >= sys.klevel:
            largerThanK = largerThanK + 1

    if len(sys.klevels) < 80:
        FP_our = FP_our + 1
    if len(sys.fixed_klevels) < 80:
        FP_fixed = FP_fixed + 1

    del sys
    print("end iteration")
    del attack_duration
    del attack
    del exp
    del fixed_detector
    del fixed_tao
    del detector
    del tao

print("FP_our", FP_our)
print("ourLength", ourLength)
print("FP_our_rate", FP_our / ourLength)
print("FP_fixed", FP_fixed)
print("totalLength", totalLength)
print("FP_fixed / totalLength", FP_fixed / totalLength)
print("largerThanK", largerThanK)
print("largerThanK / totalLength", largerThanK / totalLength)
