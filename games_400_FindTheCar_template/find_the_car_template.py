#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """
    # QHACK #

    # We perform a Deutsch-Jozsa algorithm to know if the car is behind
    # a given pair of doors.
    @qml.qnode(dev)
    def circuit1():
        # QHACK #
        # Check if the car is behind either door |00> or door |10>.
        qml.Hadamard(wires=0)
        qml.PauliX(wires=["sol"])
        qml.Hadamard(wires=["sol"])
        oracle()
        qml.Hadamard(wires=0)
        # QHACK #
        return qml.sample(wires=[0, 1])

    @qml.qnode(dev)
    def circuit2():
        # QHACK #
        # Check if the car is behind either door |00> or door |01>.
        qml.Hadamard(wires=1)
        qml.PauliX(wires=["sol"])
        qml.Hadamard(wires=["sol"])
        oracle()
        qml.Hadamard(wires=1)
        # QHACK #
        return qml.sample(wires=[0, 1])

    sol1 = circuit1()
    sol2 = circuit2()

    # process sol1 and sol2 to determine which door the car is behind.
    isin1 = any(sol1 != 0)  # is the car in |00> or |10>
    isin2 = any(sol2 != 0)  # is the car in |00> or |01>
    if not isin1 and not isin2:
        return 3
    elif isin1 and isin2:
        return 0
    elif isin1:
        return 2
    else:
        return 1
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)
        qml.Toffoli(wires=[0, 1, "sol"])
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)

    output = find_the_car(oracle)
    print(f"{output}")
