from simulators.linear.rlc_circuit_lqr import RlcCircuit
import numpy as np
from utils.attack import Attack

class rlc_circuit_bias:
    name = 'rlc_circuit'
    max_index = 1000
    dt = 0.02
    ref = [np.array([3])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.01}
        },
        'measurement': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.01}
        }
    }
    model = RlcCircuit('test', dt, max_index, noise)
    attack_start_index = 500
    bias = np.array([-0.6, 0, 0, 0])
    attack = Attack('bias', bias, attack_start_index)