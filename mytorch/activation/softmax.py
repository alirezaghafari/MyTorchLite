import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    "Compute Softmax activation function"
    exp_x = x.exp()
    sum_exp_x = exp_x @ np.ones((exp_x.shape[-1], 1))
    return exp_x * sum_exp_x ** -1
