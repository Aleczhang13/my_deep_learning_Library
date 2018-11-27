'''

loss function
'''
import numpy as np

from aleccb.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    '''
    求平方方差
    '''
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        loss = np.sum((predicted-actual)**2)
        return loss

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

