import numpy as np

from aleccb.train import train
from aleccb.nn import NeuralNet
from aleccb.layers import Linear, Tanh
from typing import List

def fiz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 3 == 0:
        return [0, 0, 1, 0]
    elif x % 5 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
