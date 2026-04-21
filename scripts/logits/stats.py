from loomlib import StatMoments, LogitSpace
import torch

B, N = 4, 12
logits = torch.randn(B, N)
processor = LogitSpace()

# Setup value range for subspace
sub_lb, sub_ub = 5, 10
pdist = torch.arange(sub_ub - sub_lb)

moments = StatMoments(pdist)

# Register processing functions
processor[sub_lb:sub_ub] = lambda x: moments.mean(torch.softmax(x, -1))
processor[sub_lb:sub_ub] = lambda x: moments.variance(torch.softmax(x, -1))
processor[sub_lb:sub_ub] = lambda x: moments.skewness(torch.softmax(x, -1))
processor[sub_lb:sub_ub] = lambda x: moments.kurtosis(torch.softmax(x, -1))

results = processor(logits)

print(f"Mean:     {results[0]}")
print(f"Variance: {results[1]}")
print(f"Skewness: {results[2]}")
print(f"Kurtosis: {results[3]}")
