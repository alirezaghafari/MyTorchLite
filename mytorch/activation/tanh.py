import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    Compute the hyperbolic tangent function (tanh)
    """
    exp_x = x.exp()
    neg_exp_x = (-x).exp()
    return (exp_x - neg_exp_x) * (exp_x + neg_exp_x) ** -1
