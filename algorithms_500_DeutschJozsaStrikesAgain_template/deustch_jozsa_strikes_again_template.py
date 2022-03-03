#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #

    dev = qml.device("default.qubit", wires=8, shots=1)

    input_wires = [0, 1]
    oracle_input_wires = [2, 3]
    oracle_output_wires = [4]
    output_wires = [5]
    all_wires = input_wires + oracle_input_wires + oracle_output_wires + output_wires

    def oracle(fs):
        """Implement the oracle described in the problem file."""
        # Usual initialisation for DJ.
        for i in oracle_input_wires:
            qml.Hadamard(wires=i)
        qml.PauliX(wires=oracle_output_wires)
        qml.Hadamard(wires=oracle_output_wires)

        for j, func in enumerate(fs):
            j_str = bin(j)[2:].zfill(2)
            # the control qubits should be in the state |j> to apply func.
            # because we can only control on |1> qubits, we flip the qubits
            # that are expected to be |0> and apply our controlled operation.
            # these qubits are flipped back after the controlled operation.
            for i, bit in enumerate(j_str):
                if bit == "0":
                    qml.PauliX(input_wires[i])
            # Controlled operation
            qml.ctrl(func, control=input_wires)(
                wires=oracle_input_wires + oracle_output_wires
            )
            # Flipping back.
            for i, bit in enumerate(j_str):
                if bit == "0":
                    qml.PauliX(input_wires[i])

        # Typical end of circuit for DJ algorithm.
        for i in oracle_input_wires:
            qml.Hadamard(wires=i)

        # Now:
        # - if all the qubits in oracle_input_wires are in the |0> state it means
        #   that func is constant.
        # - else func is balanced.
        # To test that, we flip all the qubits, meaning that if func is constant,
        # all the qubits end up in the |1> state.
        for i in oracle_input_wires:
            qml.PauliX(wires=i)
        # Then we do a Toffoli, if all the qubits are in the |1> state, the
        # output qubit is flipped and so (func_i constant) <=> (output == |1>)
        qml.Toffoli(wires=oracle_input_wires + output_wires)
        # We want to uncompute everything except the previous toffoli in order
        # to have the right answer...
        for i in oracle_input_wires:
            qml.PauliX(wires=i)

        # Uncompute
        for i in oracle_input_wires:
            qml.Hadamard(wires=i)
        for j, func in enumerate(fs):
            j_str = bin(j)[2:].zfill(2)
            for i, bit in enumerate(j_str):
                if bit == "0":
                    qml.PauliX(input_wires[i])
            qml.ctrl(func, control=input_wires)(
                wires=oracle_input_wires + oracle_output_wires
            )
            for i, bit in enumerate(j_str):
                if bit == "0":
                    qml.PauliX(input_wires[i])
        for i in oracle_input_wires:
            qml.Hadamard(wires=i)
        qml.Hadamard(wires=oracle_output_wires)
        qml.PauliX(wires=oracle_output_wires)

    @qml.qnode(dev)
    def circuit(fs):
        # Insert any pre-oracle processing here
        for i in input_wires:
            qml.Hadamard(wires=i)
        qml.PauliX(wires=output_wires)
        qml.Hadamard(wires=output_wires)

        oracle(fs)  # DO NOT MODIFY this line

        # Insert any post-oracle processing here
        for i in input_wires:
            qml.Hadamard(wires=i)
        # QHACK #

        return qml.sample(wires=input_wires)

    sample = circuit(fs)
    # From `sample` (a single call to the circuit), determine whether the function is constant or balanced.
    return "4 same" if all(sample == 0) else "2 and 2 "

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
