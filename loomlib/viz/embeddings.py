from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from . import _require_matplotlib, _transparent

if TYPE_CHECKING:
    import matplotlib.figure
    from ..encoder.batch import LoomBatch


def plot_embedding_norms(
    encoder_output: torch.Tensor,
    batch: LoomBatch,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> matplotlib.figure.Figure:
    """Heatmap of per-token L2 embedding norms from the encoder output.

    Padding positions (from ``batch.padding_mask``) are masked to white.
    This gives a quick overview of embedding magnitudes across the batch.

    Args:
        encoder_output: Tensor of shape ``[B, N_max, d_model]`` from
            ``LoomEncoder.forward()``.
        batch: The ``LoomBatch`` used to produce the output (for the
            padding mask).
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    norms = encoder_output.detach().cpu().norm(dim=-1).numpy().copy()
    mask = batch.padding_mask.detach().cpu().numpy()
    norms[mask] = np.nan

    B, N = norms.shape

    if ax is None:
        w = max(6.0, N * 0.5)
        h = max(2.0, B * 0.4 + 1.0)
        fig, ax = plt.subplots(figsize=figsize or (w, h))
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    im = ax.imshow(
        norms,
        aspect="auto",
        cmap=cmap_obj,
        interpolation="nearest",
    )

    ax.set_xlabel("token position")
    ax.set_ylabel("sequence")
    ax.set_title("Encoder Embedding Norms")

    fig.colorbar(im, ax=ax, shrink=0.8, label="L2 norm")
    fig.tight_layout()
    _transparent(fig)
    return fig
