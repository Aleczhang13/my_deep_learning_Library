'''
神经层类似于多个层的相加
'''

from typing import Sequence,Iterator,Tuple

from aleccb.layers import Layer
from aleccb.tensor import Tensor

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs:Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self,grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.params[name]
                yield param,grad






