import numpy as np
from neural_network.layers.layer import Layer

class Flatten(Layer):
    name = "Flatten"

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def parameters(self):
        return []

    def gradients(self):
        return []