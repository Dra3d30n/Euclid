import numpy as np
import neural_network.activations.activation as activation


class Tanh(activation.Activation):
    name = "Tanh"

    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x):
        y = np.tanh(x)
        return 1 - y**2