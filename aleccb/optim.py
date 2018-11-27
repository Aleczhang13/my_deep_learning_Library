'''
使用优化层进行调整我们的参数
'''
from aleccb.nn import NeuralNet

class Optimizer:
    def step(self, net:NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet):
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

