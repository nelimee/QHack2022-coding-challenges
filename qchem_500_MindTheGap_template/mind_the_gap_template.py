import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.
    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian
    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """
    # QHACK #
    hf_state = np.array([1, 1, 0, 0])
    nqubits = 4

    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(nqubits))
        return qml.expval(H)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    max_iterations = 100
    conv_tol = 1e-06
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]
    angle = [theta]
    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energy.append(cost_fn(theta))
        angle.append(theta)
        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break

    @qml.qnode(dev)
    def get_state():
        circuit(angle[-1], wires=range(nqubits))
        return qml.state()

    state_return = get_state()
    return energy[-1], state_return
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.
    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()
    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """
    # QHACK #
    ground_state2 = np.expand_dims(ground_state, axis=1)
    obs = ground_state2 * np.conj(ground_state2).T
    obs *= beta
    coeffs, obs_list = qml.utils.decompose_hamiltonian(obs)
    H2 = qml.Hamiltonian(coeffs, obs_list)
    H1 = H + H2
    return H1
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.
    Args:
        - H1 (qml.Observable): result of create_H1
    Returns:
        - (float): The excited state energy
    """
    # QHACK #
    hf_state = np.array([1, 1, 0, 0])
    nqubits = 4

    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.DoubleExcitation(param[0], wires=[0, 1, 2, 3])
        qml.SingleExcitation(param[1], wires=[0, 2])
        qml.SingleExcitation(param[2], wires=[1, 3])

    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(nqubits))
        return qml.expval(H1)

    opt = qml.GradientDescentOptimizer(stepsize=0.05)
    max_iterations = 300
    conv_tol = 1e-10
    theta = np.ones(3, requires_grad=True)
    energy = [cost_fn(theta)]
    angle = [theta]
    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)
        energy.append(cost_fn(theta))
        angle.append(theta)
        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
    return np.real(energy[-1])
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)
    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)
    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)
    answer = [np.real(E0), E1]
    print(*answer, sep=",")
