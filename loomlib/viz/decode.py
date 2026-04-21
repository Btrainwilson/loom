from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from . import _require_matplotlib, _transparent
from .colors import BLUE, GREEN, CYAN

if TYPE_CHECKING:
    import matplotlib.figure
    from ..head.loom_head import LoomHead


def _field_type_name(head: LoomHead, field_key: str) -> str | None:
    """Look up the LoomType class name for a decoded field key."""
    for entry in head.allocation.all_entries():
        short = entry.name.split(".")[-1]
        if field_key in (entry.name, short):
            return type(entry.thunk_type).__name__
    return None


def _get_entry(head: LoomHead, field_key: str):
    for entry in head.allocation.all_entries():
        short = entry.name.split(".")[-1]
        if field_key in (entry.name, short):
            return entry
    return None


def plot_decoded(
    head: LoomHead,
    z: torch.Tensor,
    *,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Per-field subplots of decoded values across a batch.

    Visualization style adapts to the field's ``LoomType``:

    - **Categorical**: bar chart of class counts.
    - **Boolean**: stacked proportion bar (True / False).
    - **Continuous** (``ContinuousScalar``, ``Scalar``, ``BitInteger``):
      strip plot of decoded values, with the type's range shaded when
      bounded.

    Args:
        head: A compiled ``LoomHead``.
        z: Logit tensor of shape ``(batch, total_logits)`` or
            ``(total_logits,)``.
        figsize: Figure size ``(width, height)`` if creating a new figure.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if z.dim() == 1:
        z = z.unsqueeze(0)

    decoded = head.decode(z.detach())

    fields = [
        k for k in decoded
        if k != "__opcode__"
    ]
    if not fields:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No fields to plot", ha="center", va="center",
                transform=ax.transAxes)
        _transparent(fig)
        return fig

    n = len(fields)
    default_w = max(2.5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize or (default_w, 3.5))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, fields):
        vals = decoded[key]
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu()
        type_name = _field_type_name(head, key)
        entry = _get_entry(head, key)

        if type_name == "Categorical":
            _plot_categorical(ax, vals, key, entry)
        elif type_name == "Boolean":
            _plot_boolean(ax, vals, key)
        else:
            _plot_continuous(ax, vals, key, type_name, entry)

    fig.suptitle("Decoded Values", fontsize=11, y=1.02)
    fig.tight_layout()
    _transparent(fig)
    return fig


def _plot_categorical(ax, vals: torch.Tensor, key: str, entry):
    n_classes = entry.thunk_type._n if entry is not None else int(vals.max().item()) + 1
    counts = torch.zeros(n_classes)
    for v in vals.flatten().long():
        if 0 <= v < n_classes:
            counts[v] += 1
    ax.bar(range(n_classes), counts.numpy(), color=BLUE, edgecolor="white")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title(key, fontsize=9)
    ax.set_xticks(range(n_classes))


def _plot_boolean(ax, vals: torch.Tensor, key: str):
    total = vals.numel()
    n_true = vals.sum().item()
    n_false = total - n_true
    ax.barh(
        [0],
        [n_true],
        color=GREEN,
        edgecolor="white",
        label="True",
        height=0.5,
    )
    ax.barh(
        [0],
        [n_false],
        left=[n_true],
        color=CYAN,
        edgecolor="white",
        label="False",
        height=0.5,
    )
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_title(key, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")


def _plot_continuous(ax, vals: torch.Tensor, key: str, type_name, entry):
    vals_np = vals.float().flatten().numpy()
    x_jitter = [0] * len(vals_np)

    ax.scatter(x_jitter, vals_np, alpha=0.6, s=18, color=BLUE,
               edgecolors="white", linewidths=0.3, zorder=3)

    if entry is not None and type_name == "ContinuousScalar":
        lo = entry.thunk_type._lo
        hi = entry.thunk_type._hi
        ax.axhspan(lo, hi, alpha=0.1, color=GREEN, zorder=1)
        ax.axhline(lo, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(hi, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xticks([])
    ax.set_title(key, fontsize=9)
    ax.set_ylabel("value")
