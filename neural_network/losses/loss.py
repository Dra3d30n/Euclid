import numpy as np

class Loss:
    name = "Loss"

    def forward(self, prediction, target):

        raise NotImplementedError

    def backward(self, prediction, target):

        raise NotImplementedError

    def __call__(self, prediction, target):
        return self.forward(prediction, target)