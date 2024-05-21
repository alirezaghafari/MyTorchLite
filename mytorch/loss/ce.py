from mytorch import Tensor
from ..activation import softmax
import numpy as np

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "Compute Categorical Cross Entropy loss"
    log_preds = preds.log()
    loss = (label * log_preds).sum().__neg__()
    return loss                                                                                 
