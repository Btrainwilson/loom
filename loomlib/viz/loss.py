from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import _require_matplotlib

if TYPE_CHECKING:
    import matplotlib.figure
    from ..slice.allocation import SliceAllocation


_TYPE_COLORS = {
    "Categorical": "#4C72B0",
    "ContinuousScalar": "#55A868",
    "BitInteger": "#DD8452",
    "Boolean": "#C44E52",
    "Scalar": "#8172B3",
}
_DEFAULT_COLOR = "#8C8C8C"


def plot_loss_breakdown(
    breakdown: dict[str, torch.Tensor],
    *,
    allocation: SliceAllocation | None = None,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Horizontal bar chart of per-field losses.

    Args:
        breakdown: The ``dict[str, Tensor]`` returned as the second element
            of ``LoomHead.loss()``.
        allocation: Optional ``SliceAllocation`` used to color-code bars by
            ``LoomType`` class.
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    names = list(breakdown.keys())
    values = [breakdown[n].detach().cpu().item() for n in names]

    sorted_pairs = sorted(zip(values, names), reverse=True)
    values = [v for v, _ in sorted_pairs]
    names = [n for _, n in sorted_pairs]

    type_map: dict[str, str] = {}
    if allocation is not None:
        for entry in allocation.all_entries():
            short = entry.name.split(".")[-1]
            type_name = type(entry.thunk_type).__name__
            type_map[entry.name] = type_name
            type_map[short] = type_name

    colors = []
    for name in names:
        type_name = type_map.get(name)
        colors.append(_TYPE_COLORS.get(type_name, _DEFAULT_COLOR))

    if ax is None:
        h = max(2.0, len(names) * 0.4 + 0.8)
        fig, ax = plt.subplots(figsize=figsize or (7, h))
    else:
        fig = ax.figure

    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("loss")
    ax.set_title("Loss Breakdown by Field")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
