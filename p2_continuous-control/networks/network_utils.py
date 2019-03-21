import numpy as np

def hidden_init(pytorch_layer):
    fan_in = pytorch_layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)