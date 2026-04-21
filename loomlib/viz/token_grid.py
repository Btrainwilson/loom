from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from . import _require_matplotlib, _transparent

if TYPE_CHECKING:
    import matplotlib.figure
    from ..encoder.loom_encoder import LoomEncoder
    from ..encoder.batch import LoomBatch


def plot_token_grid(
    encoder: LoomEncoder,
    batch: LoomBatch,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
) -> matplotlib.figure.Figure:
    """Heatmap of encoded scalar values across the batch.

    Rows are batch elements, columns are token positions.  Padding
    positions are masked to gray.  Instance boundaries from the first
    sequence are overlaid as dashed vertical lines, and x-axis labels
    show field names (from the first sequence).

    Args:
        encoder: A compiled ``LoomEncoder`` (for field-name resolution).
        batch: A ``LoomBatch`` produced by ``encoder.collate()``.
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    from .batch import _build_field_name_map

    B, N = batch.type_ids.shape
    field_map = _build_field_name_map(encoder)

    vals = batch.values.detach().cpu().numpy().copy()
    mask = batch.padding_mask.detach().cpu().numpy()
    vals[mask] = np.nan

    if ax is None:
        w = max(6.0, N * 0.5)
        h = max(2.0, B * 0.4 + 1.0)
        fig, ax = plt.subplots(figsize=figsize or (w, h))
    else:
        fig = ax.figure

    real_vals = vals[~np.isnan(vals)]
    vmax = max(abs(real_vals.min()), abs(real_vals.max()), 1e-6) if real_vals.size > 0 else 1.0

    cmap_obj = plt.get_cmap(cmap).copy()
    from .colors import PADDING_COLOR
    cmap_obj.set_bad(color=PADDING_COLOR)

    im = ax.imshow(
        vals,
        aspect="auto",
        cmap=cmap_obj,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # Instance boundary lines (from first sequence)
    prev_inst = -1
    for t in range(N):
        if mask[0, t]:
            continue
        inst_id = batch.inst_ids[0, t].item()
        if inst_id != prev_inst and prev_inst >= 0:
            ax.axvline(t - 0.5, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.6)
        prev_inst = inst_id

    # X-axis labels from first sequence
    tick_labels = []
    for t in range(N):
        if mask[0, t]:
            tick_labels.append("")
        else:
            gid = batch.field_ids[0, t].item()
            _, fname = field_map[gid]
            tick_labels.append(fname)

    ax.set_xticks(range(N))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("sequence")
    ax.set_title("Token Value Grid")

    fig.colorbar(im, ax=ax, shrink=0.8, label="encoded value")
    fig.tight_layout()
    _transparent(fig)
    return fig
