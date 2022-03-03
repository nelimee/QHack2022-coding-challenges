import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires) 

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(angles, inputs):
        qml.BasisEmbedding(features=inputs, wires=range(num_wires))
        qml.StronglyEntanglingLayers(weights=angles, wires=range(num_wires))
        return qml.expval(qml.PauliZ(0))

    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias

    # Define a cost function below with your needed arguments
    def cost(weights, bias, X, Y):

        # QHACK #
        
        # Insert an expression for your model predictions here
        predictions = [variational_classifier(weights, bias, x) for x in X]

        # QHACK #

        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # # optimize your circuit here
    epochs = 100
    shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=num_wires)
    np.random.seed(0)
    weights_init = np.random.random(size=shape, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    opt = qml.optimize.AdamOptimizer(0.1, beta1=0.9, beta2=0.999)
    print(cost(weights_init, bias_init, ising_configs, labels))
    predictions = [np.sign(variational_classifier(weights_init, bias_init, x)) for x in ising_configs]
    print(accuracy(labels, predictions))

    weights = weights_init
    bias = bias_init
    for i in range(epochs):
        weights, bias, _, _ = opt.step(cost, weights, bias, ising_configs, labels)

        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in ising_configs]
        acc = accuracy(labels, predictions)
        print(i+1, acc)
        if acc > 0.92:
            break

    # # # QHACK #

    # return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    #print(*predictions, sep=",")
