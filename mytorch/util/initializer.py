import numpy as np

def xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(1/shape[0], dtype=np.float64)

def he_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2/shape[0], dtype=np.float64)

def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(np.float64)


def zero_initializer(shape):
    return np.zeros(shape, dtype=np.float64)


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)

def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
