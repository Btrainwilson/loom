from __future__ import annotations

from typing import TYPE_CHECKING

from . import _require_matplotlib

if TYPE_CHECKING:
    import matplotlib.figure
    from ..head.loom_head import LoomHead
    from ..slice.allocation import SliceAllocation


_BRANCH_PALETTES = [
    "#4C72B0",  # steel blue
    "#DD8452",  # muted orange
    "#55A868",  # sage green
    "#C44E52",  # brick red
    "#8172B3",  # muted purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
    "#CCB974",  # olive
    "#64B5CD",  # sky blue
]
_OPCODE_COLOR = "#B0B0B0"


def _resolve_allocation(
    head_or_alloc: LoomHead | SliceAllocation,
) -> SliceAllocation:
    from ..head.loom_head import LoomHead
    from ..slice.allocation import SliceAllocation

    if isinstance(head_or_alloc, LoomHead):
        return head_or_alloc.allocation
    if isinstance(head_or_alloc, SliceAllocation):
        return head_or_alloc
    raise TypeError(
        f"Expected LoomHead or SliceAllocation, got {type(head_or_alloc).__name__}"
    )


def plot_allocation(
    head_or_alloc: LoomHead | SliceAllocation,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Horizontal stacked-bar showing the logit vector partitioned by branch.

    Each ``SliceEntry`` is drawn as a colored rectangle spanning its logit
    range, labeled with the short field name and type.

    Args:
        head_or_alloc: A ``LoomHead`` or ``SliceAllocation``.
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    alloc = _resolve_allocation(head_or_alloc)

    branch_names = list(alloc.branches.keys())
    has_opcode = alloc.opcode is not None
    row_labels: list[str] = []
    if has_opcode:
        row_labels.append("opcode")
    row_labels.extend(
        name if name != "__root__" else "fields" for name in branch_names
    )

    if ax is None:
        height = max(1.2, 0.6 * len(row_labels) + 0.6)
        w = max(6.0, alloc.total_logits * 0.35)
        fig, ax = plt.subplots(figsize=figsize or (w, height))
    else:
        fig = ax.figure

    color_map: dict[str, str] = {}
    for i, name in enumerate(branch_names):
        color_map[name] = _BRANCH_PALETTES[i % len(_BRANCH_PALETTES)]

    y_positions = {label: i for i, label in enumerate(row_labels)}

    bar_height = 0.6

    def _draw_entry(entry, row_label, color):
        y = y_positions[row_label]
        width = entry.end - entry.start
        rect = mpatches.FancyBboxPatch(
            (entry.start, y - bar_height / 2),
            width,
            bar_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        short_name = entry.name.split(".")[-1]
        type_label = type(entry.thunk_type).__name__
        label = f"{short_name}\n{type_label}" if width >= 3 else short_name
        ax.text(
            entry.start + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=max(6, min(9, width * 1.5)),
            color="white",
            fontweight="bold",
        )

    if has_opcode:
        _draw_entry(alloc.opcode, "opcode", _OPCODE_COLOR)

    for branch_name, branch in alloc.branches.items():
        row_label = branch_name if branch_name != "__root__" else "fields"
        color = color_map[branch_name]
        for entry in branch.entries:
            _draw_entry(entry, row_label, color)

    ax.set_xlim(-0.5, alloc.total_logits + 0.5)
    ax.set_ylim(-0.5, len(row_labels) - 0.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("logit index")
    ax.set_title(f"Logit Allocation ({alloc.total_logits} total)")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
