from simulators.linear.rlc_circuit import RlcCircuit
from simulators.linear.platoon import Platoon
from simulators.linear.F16 import F16
from simulators.linear.boeing747 import Boeing
from simulators.linear.aircraft_pitch import AircraftPitch
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
            'type': 'box_uniform',
            'param': {'lo': np.ones(2) * -0.02,
                      'up': np.ones(2) * 0.02}
        }
    }
    model = RlcCircuit('test', dt, max_index, noise)
    attack_start_index = 400
    # bias = np.array([0, 0, 0, 0])
    # attack = Attack('bias', bias, attack_start_index)


class F16:
    name = 'F16'
    max_index = 500
    dt = 0.02
    ref = [np.array([0.0872665 * 57.3])] * 501
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(4) * -0.00001,
                      'up': np.ones(4) * 0.00001}
        }
    }
    model = F16('test', dt, max_index, noise)
    attack_start_index = 400

class Boeing:
    name = 'Boeing'
    max_index = 500
    dt = 0.02
    ref = [np.array([15])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(5) * -0.001,
                      'up': np.ones(5) * 0.001}
        }
    }
    model = Boeing('test', dt, max_index, noise)
    attack_start_index = 400

class AircraftPitch:
    name = 'AircraftPitch'
    max_index = 1500
    dt = 0.02
    ref = [np.array([0.2])] * 1501
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.ones(3) * -0.0001,
                      'up': np.ones(3) * 0.0001}
        }
    }
    model = AircraftPitch('test', dt, max_index, noise)
    attack_start_index = 1500


class Platoon:
    name = 'platoon'
    max_index = 400
    dt = 0.04
    # ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    ref = [np.array([1])] * 201
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
    # attack_start_index = 400        # for time-consuming
    # attack_start_index = 55     # for three detector
    attack_start_index = 65
