#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        for q in range(3):
            qml.Hadamard(wires=q)
        for i in range(2**3):
            istr = bin(i)[2:].zfill(3)
            for j, s in enumerate(istr):
                if s == "0":
                    qml.PauliX(wires=j)
            qml.ctrl(qml.RY, control=[0, 1, 2])(thetas[i], wires=3)
            for j, s in enumerate(istr):
                if s == "0":
                    qml.PauliX(wires=j)
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
