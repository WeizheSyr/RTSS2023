from simulators.linear.rlc_circuit_lqr import RlcCircuit
from simulators.linear.platoon import Platoon
import numpy as np
from utils.attack import Attack

class rlc_circuit_bias:
    name = 'rlc_circuit'
    max_index = 500
    dt = 0.02
    # ref = [np.array([2])] * (max_index + 1)
    ref = [np.array([2])] * 201 + [np.array([2])] * 200 + [np.array([2])] * 100
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.0001}
        },
        'measurement': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.0001}
        }
    }
    model = RlcCircuit('test', dt, max_index, noise)
    attack_start_index = 200
    # bias = np.array([0, 0, 0, 0])
    # attack = Attack('bias', bias, attack_start_index)

class Platoon:
    name = 'platoon'
    max_index = 400
    dt = 0.04
    # ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    ref = [np.array([0.5])] * 401
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(7) * -0.002,
                      'up': np.ones(7) * 0.002}
        },
        'measurement': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(7) * -0.00,
                      'up': np.ones(7) * 0.00}
        }
    }
    model = Platoon('test', dt, max_index, noise)
    attack_start_index = 400        # for time-consuming
    # attack_start_index = 55     # for three detector
