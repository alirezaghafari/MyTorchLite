from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np


class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "Implement forward pass"
        output = x @ self.weight
        if self.need_bias:
            output = output + self.bias
        return output

    def initialize(self):
        "Initialize weight by initializer function (mode)"
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), mode=self.initialize_mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode="zero"),
                requires_grad=True
            )

    def zero_grad(self):
        "Implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "Return weights and bias"
        return [self.weight, self.bias] if self.need_bias else [self.weight]


    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs,
                                                                   self.outputs)
