import numpy as np
import neural_network.activations.activation as activation

class ReLU(activation.Activation):
    name="ReLU"
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    @staticmethod
    def backward(x):
        return (x > 0).astype(float)
    