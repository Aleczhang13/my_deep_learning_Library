"""
我们进行训练
"""
import numpy as np

from aleccb.train import train
from aleccb.nn import NeuralNet
from aleccb.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

net = NeuralNet([
    Linear(input_size=2, out_size=2),
    Tanh(),
    Linear(input_size=2, out_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)

