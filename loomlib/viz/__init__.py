"""Visualization helpers for Loom schemas, heads, encoders, and logit vectors.

Install with ``pip install loomlib[viz]`` to pull in matplotlib.
"""

from __future__ import annotations


def _require_matplotlib():
    """Raise a helpful error if matplotlib is not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "loomlib.viz requires matplotlib.  "
            "Install it with:  pip install loomlib[viz]"
        ) from None


def _transparent(fig) -> None:
    """Make *fig* and all its axes transparent so plots blend with any page background."""
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)


# Decoder visualizations
from .allocation import plot_allocation
from .logits import plot_logits
from .loss import plot_loss_breakdown
from .decode import plot_decoded

# Encoder visualizations
from .batch import plot_batch
from .token_grid import plot_token_grid
from .embeddings import plot_embedding_norms

__all__ = [
    # Decoder
    "plot_allocation",
    "plot_logits",
    "plot_loss_breakdown",
    "plot_decoded",
    # Encoder
    "plot_batch",
    "plot_token_grid",
    "plot_embedding_norms",
]
