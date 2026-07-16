import numpy as np 

class Activation:
    name = ""
    @staticmethod
    def forward(x):
        return x
    
    @staticmethod
    def backward(x):
        return np.ones_like(x)