import torch


def greedy_fill_left(B: torch.Tensor, D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy left-to-right allocation.

    Args:
        B: ``(batch, n)`` or ``(n,)`` upper bounds.
        D: ``(batch,)`` or scalar totals.

    Returns:
        ``(alloc, frac)`` where *alloc* sums to *D* along dim-1
        and *frac* = alloc / B (zero where B is zero).
    """
    B = torch.as_tensor(B, dtype=torch.float)
    if B.dim() == 1:
        B = B.unsqueeze(0)
    batch, n = B.shape

    D = torch.as_tensor(D, dtype=torch.float)
    if D.dim() == 0:
        D = D.expand(batch)
    elif D.dim() == 1:
        D = D.expand(batch)
    else:
        raise ValueError("D must be scalar or 1D batch vector")

    cum_B = torch.cumsum(B, dim=1)
    left = D.unsqueeze(1) - torch.cat([
        torch.zeros((batch, 1), device=B.device, dtype=B.dtype),
        cum_B[:, :-1],
    ], dim=1)
    left = torch.clamp(left, min=0)
    alloc = torch.min(B, left)

    over = alloc.sum(dim=1) - D
    for i in range(batch):
        if over[i] > 1e-7:
            nz = (alloc[i] > 0).nonzero(as_tuple=True)[0]
            if len(nz) > 0:
                alloc[i, nz[-1]] -= over[i]

    p = alloc / B
    p = torch.where(B == 0, torch.zeros_like(p), p)

    if alloc.shape[0] == 1:
        alloc = alloc[0]
        p = p[0]
    return alloc, p
