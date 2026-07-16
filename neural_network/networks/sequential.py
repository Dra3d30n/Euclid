import numpy as np
from neural_network.networks.network import Network

class Sequential(Network):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.gradients())
        return grads