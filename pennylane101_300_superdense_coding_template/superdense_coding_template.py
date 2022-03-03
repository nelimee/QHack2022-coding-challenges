#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def superdense_coding(bits, alpha):
    """Construct a quantum circuit that implements superdense coding, given a not necessarily maximally entangled state

    Args:
        - bits (int): 0 (binary: 00), 1 (binary: 01), 2 (binary: 10), or 3 (binary: 11), Alice's bits that she wants to communicate to Bob.
        - alpha (float): angle parametrizing the entangled state

    Returns:
        - (np.tensor): Probability that Bob will guess Alice's bits correctly
    """

    # QHACK #

    # Prepare entangled state here
    qml.RY(2 * alpha, wires=0)
    qml.CNOT(wires=[0, 1])

    # Implement Alice's operations on her qubit here
    if bits == 0:
        pass
    elif bits == 1:
        qml.PauliX(wires=0)
    elif bits == 2:
        qml.PauliZ(wires=0)
    elif bits == 3:
        qml.PauliX(wires=0)
        qml.PauliZ(wires=0)

    # Implement Bob's measurement procedure here
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)

    # QHACK #

    return qml.probs(wires=[0, 1])


def return_probs(bits, alpha):
    """Returns the output of the superdense_coding function for a given index (bits)"""
    # DO NOT MODIFY anything in this code block
    return superdense_coding(bits, alpha)[bits].numpy()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = return_probs(int(inputs[0]), float(inputs[1]))
    print(f"{output:.6f}")
