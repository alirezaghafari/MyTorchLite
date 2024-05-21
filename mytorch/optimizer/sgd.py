from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # Update weights and biases using gradient descent
        for layer in self.layers:
            # Update weights if gradients exist
            if layer.weight is not None and layer.weight.grad is not None:
                layer.weight = layer.weight - layer.weight.grad * self.learning_rate
            # Update biases if gradients exist and bias is needed
            if layer.need_bias and layer.bias is not None and layer.bias.grad is not None:
                layer.bias = layer.bias - layer.bias.grad * self.learning_rate
