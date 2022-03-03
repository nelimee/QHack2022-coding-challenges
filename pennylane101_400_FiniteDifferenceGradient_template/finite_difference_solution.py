#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)

""" Coding the shift rule yourself where you need to run for each unit vector e_i (hence for each parameter indexed i) 0.5*(f(x+ pi/2) - f(x-pi/2))"""

def my_finite_diff_grad(params):
    """Function that returns the gradients of the cost function (defined below) with respect 
    to all parameters in params.

    Args:
        - params (np.ndarray): The parameters needed to create the variational circuit.

    Returns:
        - gradients (np.ndarray): the gradient w.r.t. each parameter
    """

    gradients = np.zeros([len(params)])
    for i in range(len(params)):
        # QHACK #
        delta_plus = params.copy()
        delta_plus[i] += np.pi / 2.0
        gradients[i] = cost(delta_plus)
        delta_moins = delta_plus.copy()
        delta_moins[i] -= np.pi
        gradients[i] -= cost(delta_moins)
        gradients[i] *= 0.5


        # QHACK #

    return gradients


def variational_circuit(params):
    """A layered variational circuit. The first layer comprises of x, y, and z rotations on wires
    0, 1, and 2, respectively. The second layer is a ring of CNOT gates. The final layer comprises 
    of x, y, and z rotations on wires 0, 1, and 2, respectively.
    """

    # DO NOT MODIFY anything in this code block
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")


@qml.qnode(dev)
def cost(params):
    """A QNode that pairs the variational_circuit with an expectation value measurement.

    Args:
        - params (np.ndarray): Variational circuit parameters

    Returns:
        - (float): qml.expval(qml.PauliY(0) @ qml.PauliZ(2))
    """

    # DO NOT MODIFY anything in this code block
    variational_circuit(params)
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    params = np.array(sys.stdin.read().split(","), dtype=float)
    output = my_finite_diff_grad(params)
    print(*output, sep=",")
