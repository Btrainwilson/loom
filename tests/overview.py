import torch
import loomlib
from tensor_mosaic import Mosaic


# We begin by initializing a logitspace.
# This keeps track of logits for us.

from loomlib import Thunker

class Mean(Thunker):
    def decode(self, x):
        return torch.mean(x, dim=-1)
    def encode(self, x, mean):
        return torch.ones_like(x) * mean

mosaic = Mosaic(dim = 1)
mosaic.MEAN = 10
print(mosaic.shape)

mean = Mean()

x = torch.randn(mosaic.shape).unsqueeze(0)
print(mosaic.MEAN)
print(type(mosaic.MEAN))
y = x[:, mosaic.MEAN]
y_mean = mean.decode(y)
x_decode = mean.encode(torch.zeros_like(x), y_mean.unsqueeze(1))
print(y_mean)



