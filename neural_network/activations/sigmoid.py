import numpy as np
import neural_network.activations.activation as activation


class Sigmoid(activation.Activation):
    name = "Sigmoid"

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        y = Sigmoid.forward(x)
        return y * (1 - y)