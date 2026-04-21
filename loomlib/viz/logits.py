from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import _require_matplotlib

if TYPE_CHECKING:
    import matplotlib.figure
    from ..head.loom_head import LoomHead
    from ..slice.allocation import SliceAllocation


def plot_logits(
    head_or_alloc: LoomHead | SliceAllocation,
    z: torch.Tensor,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
) -> matplotlib.figure.Figure:
    """Heatmap of raw logit values with field boundaries overlaid.

    Args:
        head_or_alloc: A ``LoomHead`` or ``SliceAllocation``.
        z: Logit tensor of shape ``(batch, total_logits)`` or
            ``(total_logits,)``.
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.
        cmap: Matplotlib colormap name. Defaults to ``"RdBu_r"`` (diverging,
            centered at 0).

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    from .allocation import _resolve_allocation

    alloc = _resolve_allocation(head_or_alloc)

    z_np = z.detach().cpu()
    if z_np.dim() == 1:
        z_np = z_np.unsqueeze(0)
    z_np = z_np.numpy()

    entries = alloc.all_entries()

    if ax is None:
        w = max(6.0, alloc.total_logits * 0.3)
        h = max(2.0, z_np.shape[0] * 0.35 + 1.0)
        fig, ax = plt.subplots(figsize=figsize or (w, h))
    else:
        fig = ax.figure

    vmax = max(abs(z_np.min()), abs(z_np.max()), 1e-6)
    im = ax.imshow(
        z_np,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    for entry in entries:
        if entry.start > 0:
            ax.axvline(entry.start - 0.5, color="black", linewidth=0.5,
                       linestyle="--", alpha=0.5)

    tick_positions = []
    tick_labels = []
    for entry in entries:
        mid = (entry.start + entry.end - 1) / 2.0
        tick_positions.append(mid)
        tick_labels.append(entry.name.split(".")[-1])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("sample")
    ax.set_title("Logit Heatmap")

    fig.colorbar(im, ax=ax, shrink=0.8, label="logit value")
    fig.tight_layout()
    return fig
