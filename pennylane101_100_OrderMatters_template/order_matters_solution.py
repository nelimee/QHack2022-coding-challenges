#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

""" This one is quite straitghtforward from the description"""

def compare_circuits(angles):
    """Given two angles, compare two circuit outputs that have their order of operations flipped: RX then RY VERSUS RY then RX.

    Args:
        - angles (np.ndarray): Two angles

    Returns:
        - (float): | < \sigma^x >_1 - < \sigma^x >_2 |
    """

    # QHACK #

    # define a device and quantum functions/circuits here
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit_1(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuit_2(params):
        qml.RY(params[1], wires=0)
        qml.RX(params[0], wires=0)
        return qml.expval(qml.PauliX(0))

    out1 = circuit_1(angles)
    out2 = circuit_2(angles)


    return np.abs(out1-out2)

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    angles = np.array(sys.stdin.read().split(","), dtype=float)
    output = compare_circuits(angles)
    print(f"{output:.6f}")
