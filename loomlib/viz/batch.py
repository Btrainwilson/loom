from __future__ import annotations

from typing import TYPE_CHECKING

from . import _require_matplotlib
from .allocation import _BRANCH_PALETTES

if TYPE_CHECKING:
    import matplotlib.figure
    from ..encoder.loom_encoder import LoomEncoder
    from ..encoder.batch import LoomBatch

_PADDING_COLOR = "#E0E0E0"


def _build_field_name_map(encoder: LoomEncoder) -> dict[int, tuple[str, str]]:
    """Map global field id -> (branch_name, field_name)."""
    result: dict[int, tuple[str, str]] = {}
    for bname, fmap in encoder._global_field_id.items():
        for fname, gid in fmap.items():
            result[gid] = (bname, fname)
    return result


def plot_batch(
    encoder: LoomEncoder,
    batch: LoomBatch,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Per-sequence row of colored field-token blocks.

    Each real token is a colored rectangle (by branch), labeled with the
    field name and encoded scalar value.  Padding tokens are gray.
    Instance boundaries are marked with heavier borders.

    Args:
        encoder: A compiled ``LoomEncoder`` (for field-name resolution).
        batch: A ``LoomBatch`` produced by ``encoder.collate()``.
        ax: Optional matplotlib ``Axes`` to draw into.
        figsize: Figure size ``(width, height)`` if creating a new figure.

    Returns:
        The matplotlib ``Figure``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    B, N = batch.type_ids.shape
    field_map = _build_field_name_map(encoder)

    color_map: dict[str, str] = {}
    for i, name in enumerate(encoder.branch_names):
        color_map[name] = _BRANCH_PALETTES[i % len(_BRANCH_PALETTES)]

    if ax is None:
        w = max(6.0, N * 1.3)
        h = max(1.6, B * 1.0 + 0.6)
        fig, ax = plt.subplots(figsize=figsize or (w, h))
    else:
        fig = ax.figure

    cell_h = 0.7

    for b in range(B):
        prev_inst = -1
        for t in range(N):
            is_pad = batch.padding_mask[b, t].item()
            x, y = t, b

            if is_pad:
                color = _PADDING_COLOR
                label = ""
            else:
                gid = batch.field_ids[b, t].item()
                bname, fname = field_map[gid]
                color = color_map[bname]
                val = batch.values[b, t].item()
                label = f"{fname}\n{val:.2g}"

            inst_id = batch.inst_ids[b, t].item()
            edge_w = 2.0 if (not is_pad and inst_id != prev_inst and prev_inst >= 0) else 0.8
            prev_inst = inst_id if not is_pad else prev_inst

            rect = mpatches.FancyBboxPatch(
                (x + 0.02, y - cell_h / 2 + 0.02),
                0.96,
                cell_h - 0.04,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="white",
                linewidth=edge_w,
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    x + 0.5, y,
                    label,
                    ha="center", va="center",
                    fontsize=max(6, min(8, 10 - N * 0.15)),
                    color="white", fontweight="bold",
                )

    # Instance dividers
    for b in range(B):
        prev_inst = -1
        for t in range(N):
            if batch.padding_mask[b, t].item():
                continue
            inst_id = batch.inst_ids[b, t].item()
            if inst_id != prev_inst and prev_inst >= 0:
                ax.plot(
                    [t - 0.01, t - 0.01],
                    [b - cell_h / 2 - 0.05, b + cell_h / 2 + 0.05],
                    color="black", linewidth=1.2, alpha=0.6,
                )
            prev_inst = inst_id

    ax.set_xlim(-0.1, N + 0.1)
    ax.set_ylim(B - 0.5, -0.5)
    ax.set_yticks(range(B))
    ax.set_yticklabels([f"seq {b}" for b in range(B)])
    ax.set_xticks(range(N))
    ax.set_xticklabels([str(t) for t in range(N)])
    ax.set_xlabel("token position")
    ax.set_title(f"Field-Token Batch ({B} seqs, {N} max tokens)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = []
    for bname in encoder.branch_names:
        display = bname if bname != "__root__" else "fields"
        handles.append(mpatches.Patch(color=color_map[bname], label=display))
    handles.append(mpatches.Patch(color=_PADDING_COLOR, label="padding"))
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              framealpha=0.8, edgecolor="none")

    fig.tight_layout()
    return fig
