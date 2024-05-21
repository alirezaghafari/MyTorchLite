from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "Implement Mean Squared Error loss"
    num_samples = preds.shape[0] 
    squared_error = (preds - actual) ** 2  
    mean_squared_error = squared_error.sum() * num_samples ** -1 
    return mean_squared_error
