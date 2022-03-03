#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #

    wires = cnot.wires

    neighbourhoud = set(graph[wires[0]])
    explored_nodes = set([wires[1]])

    def explore_recursive(start, explored_nodes, depth: int = 0):
        """Recursively explore the graph and returns the length of the shortest path.

        This function basically performs a shortest-path between the **node** start
        and the **set of nodes** neighbourhoud.
        """
        if start in neighbourhoud:
            return depth
        neighbours = graph[start]
        depths = []
        for n in neighbours:
            if n in explored_nodes:
                continue
            explored_nodes.add(n)
            depths.append(explore_recursive(n, explored_nodes, depth + 1))
        return min(depths, default=100)

    # The number of CNOTs needed is twice the longest path (because we need to
    # uncompute).
    return 2 * explore_recursive(wires[1], explored_nodes)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
