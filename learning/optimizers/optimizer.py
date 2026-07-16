import numpy as np

class Optimizer:
    name = "Optimizer"

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def step(self, params, grads):

        raise NotImplementedError

    def zero_grad(self, model):
        for param in model.parameters():
            param.grad = 0