#! /usr/bin/python3

import sys
import numpy as np

"""For this one, we needed to expand the final state to ontain a system of equations for finding the 3 theta angles given the amplitudes a,b,c,d.
We had to use the following formulae to get a small error rate, although there are many ways to obtain the solution.

"""

def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #

    def n(theta):

        while theta < -np.pi:
            theta += 2 * np.pi
        while theta > np.pi:
            theta -= 2 * np.pi
        return theta

        
    b2plusc2 = b**2 + c**2
    b2moinsc2 = b**2 - c**2
    
    theta2 = n(-np.arccos(b2moinsc2 / b2plusc2))

    theta1 = c / np.sin(theta2 / 2.0)
    theta1 = np.arcsin(theta1) * 2.0

    a2plusd2 = a**2 + d**2
    a2moinsd2 = a**2 - d**2
    theta3 = np.sign(theta1) * np.arccos(a2moinsd2 / a2plusd2)

    return theta1, theta2, theta3
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
