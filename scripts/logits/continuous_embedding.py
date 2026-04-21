import torch
import matplotlib.pyplot as plt

# Settings
B = 2
N = 12
x = torch.randn((B, N), requires_grad=True)

# Distribution parameters
pdist = torch.arange(N).float()

# Softmax distribution
logits = x
probs = torch.softmax(logits, dim=-1)

# Calculate statistical moments
mean = torch.sum(pdist * probs, dim=-1)
variance = torch.sum(((pdist - mean.unsqueeze(-1)) ** 2) * probs, dim=-1)
skewness = torch.sum(((pdist - mean.unsqueeze(-1)) ** 3) * probs, dim=-1) / variance.pow(1.5)
kurt = torch.sum(((pdist - mean.unsqueeze(-1)) ** 4) * probs, dim=-1) / variance.pow(2)

print("Mean:", mean)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurt)

# Plotting
def plot_distribution(batch_idx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Logits
    axes[0].bar(range(N), logits[batch_idx].detach().numpy())
    axes[0].set_title(f'Logits for Batch {batch_idx}')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Logit Value')

    # Probabilities
    axes[1].bar(range(N), probs[batch_idx].detach().numpy(), color='orange')
    axes[1].set_title(f'Probabilities for Batch {batch_idx}')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Probability')

    plt.tight_layout()
    plt.show()

# Visualize each batch
for b in range(B):
    plot_distribution(b)
