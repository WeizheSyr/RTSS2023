from systemALLDim import SystemALLDim
from EvaluationFixedDetector import SystemFixedDetector
from ThreeDetector import ThreeDetector
from utils.Baseline import Platoon
from utils.detector.windowBased import window
from FPEvaluation import FPEvaluation
import matplotlib.pyplot as plt
import numpy as np
import time

# tao = np.ones(7) * 0.1
# tao = [0.5] * 7
# detector = window(tao, 7, 10)
# fixed_tao = np.ones(7) * 0.1
# fixed_detector = window(fixed_tao, 7, 10)
# noauth_detector = window(fixed_tao, 7, 10)
# exp = Platoon
# attack = np.zeros(50)
# attack_duration = 50
# attack = np.zeros(attack_duration)

# sys = ThreeDetector(detector=detector, fixed_detector=fixed_detector, noauth_detector=noauth_detector, exp=exp, attack=attack, attack_duration=attack_duration)

FP_our = 0
FP_fixed = 0
largerThanK = 0
totalLength = 0
ourLength = 0

for i in range(20):
    # rseed = np.uint32(int(time.time()))
    # print(rseed)
    # np.random.seed(rseed)
    print("iteration num: ", i)
    tao = np.ones(7) * 0.03
    detector = window(tao, 7, 10)
    fixed_tao = np.ones(7) * 0.015
    fixed_detector = window(fixed_tao, 7, 10)
    exp = Platoon
    attack = np.zeros(50)
    attack_duration = 500

    sys = FPEvaluation(detector=detector, fixed_detector=fixed_detector, exp=exp, attack=attack, attack_duration=attack_duration)
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
