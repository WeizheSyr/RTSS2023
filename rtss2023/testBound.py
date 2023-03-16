from system import System
from utils.Baseline import rlc_circuit_bias
from utils.detector import windowBased

detector = windowBased
exp = rlc_circuit_bias
attack = []
attack_duration = len(attack)

sys = System(detector=detector, exp=exp, attack=attack, attack_duration=attack_duration)
