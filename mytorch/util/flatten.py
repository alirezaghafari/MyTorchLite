import numpy as np
from mytorch import Tensor, Dependency

def flatten(x: Tensor) -> Tensor:
    data = x.data.flatten()
    req_grad = x.requires_grad
    depends_on = [Dependency(x, lambda grad: grad.reshape(x.shape))] if req_grad else []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
