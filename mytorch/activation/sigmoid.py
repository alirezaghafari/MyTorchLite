import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    "Compute Sigmoid activation function"
    return (1 + (-x).exp()) ** -1
