import torch


class StatMoments:
    """Compute statistical moments of a distribution over a point mass vector."""

    def __init__(self, pmass: torch.Tensor):
        self.pmass = pmass.float()

    def mean(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.pmass, dim=-1)

    def variance(self, probs: torch.Tensor) -> torch.Tensor:
        mean = self.mean(probs).unsqueeze(-1)
        return torch.sum(((self.pmass - mean) ** 2) * probs, dim=-1)

    def skewness(self, probs: torch.Tensor) -> torch.Tensor:
        mean = self.mean(probs).unsqueeze(-1)
        var = self.variance(probs).unsqueeze(-1)
        return torch.sum(((self.pmass - mean) ** 3) * probs, dim=-1) / (
            var.squeeze(-1).clamp(min=1e-6) ** 1.5
        )

    def kurtosis(self, probs: torch.Tensor) -> torch.Tensor:
        mean = self.mean(probs).unsqueeze(-1)
        var = self.variance(probs).unsqueeze(-1)
        return torch.sum(((self.pmass - mean) ** 4) * probs, dim=-1) / (
            var.squeeze(-1).clamp(min=1e-6) ** 2
        )
