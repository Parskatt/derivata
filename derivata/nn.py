import numpy as np
import derivata

class Linear:
    def __init__(self,in_channels,out_channels,bias=True):
        self.W = derivata.Variable(np.random.randn(out_channels,in_channels))
        if bias:
            self.b = derivata.Variable(np.random.randn(out_channels))
    def __call__(self,x):
        return self.W@x + self.b