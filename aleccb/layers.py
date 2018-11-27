'''
input -> Linear ->Tanh -> Linear -> output
'''

from aleccb.tensor import Tensor
import numpy as np
from typing import Dict, Callable
class Layer:
    def __init__(self):
        self.params: Dict[str,Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(selfs, inputs: Tensor) -> Tensor:
        '''
            produce the outputs corresponding to these input
        '''
        raise NotImplementedError
    def backward(self,grad:Tensor) -> Tensor:

        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size: int,out_size: int) -> None:
        super(Linear, self).__init__()
        self.params["w"] = np.random.rand(input_size, out_size)
        self.params["b"] = np.random.rand(out_size)

    def forward(self, inputs: Tensor) ->Tensor:
        self.inputs = inputs
        return input @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        '''
            如果有函数是 y = f(X), X = a*b + c
            dy/da = f'(x)*b
            dy/db = f'(x)*a
            dy/dc = f'(x)

            如果有函数是 y = f(X), X = a @ b + c
            dy/da = f'(x)@b.T
            dy/db = a.T@f'(x)
            dy/dc = f'(x)
        '''
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad@ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
        '''
        对输出的函数进行激活
        '''
        def __init__(self, f:  F, f_prime: F) -> None:
            super().__init__()
            self.f = f
            self.f_prime = f_prime

        def forward(self, inputs: Tensor) -> Tensor:
            self.inputs = inputs
            return self.f(inputs)

        def backward(self,grad:Tensor) -> Tensor:
            '''
            如果 if y = f(x) 和 x = g(z),此时
            dy/dz = f'(x) * g'(z)
            '''
            return self.f_prime(self.inputs) * grad



def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)












