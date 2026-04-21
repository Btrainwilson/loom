import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def randSimplex(mean, dj):
    qj = torch.nn.functional.softmax(- torch.randn(dj.shape) / dj, dim=-1)
    return mean / dj * qj

def rand_weighted_simplex(D, dj):
    n = len(dj)
    u = torch.sort(torch.cat([torch.zeros(1), torch.rand(n-1), torch.ones(1)]))[0]
    w = u[1:] - u[:-1]
    denom = torch.dot(dj, w)
    pj = D * w / denom
    return pj

def plot_marginals_rainbow(D, dj, N=10000, seed=0):
    torch.manual_seed(seed)
    n = dj.shape[0]
    samples_biased = torch.stack([randSimplex(D, dj) for _ in range(N)])
    samples_uniform = torch.stack([rand_weighted_simplex(D, dj) for _ in range(N)])

    palette = sns.color_palette("rainbow", n)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for i in range(n):
        sns.histplot(
            samples_biased[:, i].numpy(),
            bins=60, color=palette[i], label=f"$p_{i+1}$", kde=True,
            stat="density", alpha=0.7, ax=axs[0]
        )
    axs[0].set_title("randSimplex (biased)")
    axs[0].set_xlabel("$p_j$")
    axs[0].legend()

    for i in range(n):
        sns.histplot(
            samples_uniform[:, i].numpy(),
            bins=60, color=palette[i], label=f"$p_{i+1}$", kde=True,
            stat="density", alpha=0.7, ax=axs[1]
        )
    axs[1].set_title("Uniform Weighted Simplex")
    axs[1].set_xlabel("$p_j$")
    axs[1].legend()

    plt.suptitle(f"Marginal Distributions of $p_j$ for D={D}, dj={dj.tolist()}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

# Example usage
D = 7.0
dj = torch.arange(1, 6, dtype=torch.float)
plot_marginals_rainbow(D, dj, N=10000)

