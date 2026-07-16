import numpy as np
import neural_network.activations.activation as activation

class Linear(activation.Activation):
    name="Linear"
    @staticmethod
    def forward(x):
        return x
    @staticmethod
    def backward(x):
        return 1
    