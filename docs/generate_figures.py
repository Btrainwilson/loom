#!/usr/bin/env python
"""Generate static figures for the Sphinx docs using loomlib.viz.

Run from the repo root (or from docs/):

    python docs/generate_figures.py

Outputs PNGs to docs/_static/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch

# Ensure the local loomlib is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loomlib import (
    LoomModel,
    LoomUnion,
    LoomCompiler,
    Categorical,
    ContinuousScalar,
    BitInteger,
    Boolean,
    Scalar,
)
from loomlib.viz import (
    plot_allocation, plot_logits, plot_loss_breakdown, plot_decoded,
    plot_batch, plot_token_grid, plot_embedding_norms,
)

OUT_DIR = Path(__file__).resolve().parent / "_static"
OUT_DIR.mkdir(exist_ok=True)

DPI = 150
SAVEFIG_KW = dict(dpi=DPI, bbox_inches="tight", transparent=True)


# -- Schemas (same as in getting-started.md) --------------------------------

class MoveAction(LoomModel):
    direction: Categorical[4]
    speed: ContinuousScalar[0.0, 10.0]


class CastSpell(LoomModel):
    spell_id: Categorical[6]
    power: BitInteger[8]


class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell


# Encoder example schemas (scalar-encoding types)
class Steering(LoomModel):
    angle: ContinuousScalar[-1.0, 1.0]
    throttle: ContinuousScalar[0.0, 1.0]


class Shooting(LoomModel):
    active: Boolean()
    power: Scalar()


class GameAction(LoomUnion):
    steer: Steering
    shoot: Shooting


def main() -> None:
    torch.manual_seed(42)

    union_head = LoomCompiler.build_head(AgentAction, d_model=256)
    flat_head = LoomCompiler.build_head(MoveAction, d_model=128)

    # 1. Allocation map (union)
    fig = plot_allocation(union_head)
    fig.savefig(OUT_DIR / "allocation_union.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'allocation_union.png'}")

    # 2. Logit heatmap (union, small batch)
    hidden = torch.randn(6, 256)
    z = union_head(hidden)
    fig = plot_logits(union_head, z)
    fig.savefig(OUT_DIR / "logits_heatmap.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'logits_heatmap.png'}")

    # 3. Loss breakdown (union)
    targets = {
        "__opcode__": torch.tensor([0, 0, 0, 1, 1, 1]),
        "direction": torch.tensor([0, 1, 2, 0, 0, 0]),
        "speed": torch.tensor([5.0, 3.0, 7.5, 0.0, 0.0, 0.0]),
        "spell_id": torch.tensor([0, 0, 0, 3, 5, 1]),
        "power": torch.tensor([0, 0, 0, 100, 200, 50]),
    }
    _, breakdown = union_head.loss(z, targets)
    fig = plot_loss_breakdown(breakdown, allocation=union_head.allocation)
    fig.savefig(OUT_DIR / "loss_breakdown.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'loss_breakdown.png'}")

    # 4. Decoded values (flat model, larger batch for visible spread)
    hidden_flat = torch.randn(20, 128)
    z_flat = flat_head(hidden_flat)
    fig = plot_decoded(flat_head, z_flat)
    fig.savefig(OUT_DIR / "decoded_values.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'decoded_values.png'}")

    # -- Encoder figures (scalar-encoding schema) ----------------------------

    encoder = LoomCompiler.build_encoder(GameAction, d_model=64)

    enc_data = [
        [("steer", {"angle": 0.5, "throttle": 0.8}),
         ("shoot", {"active": True, "power": 3.0}),
         ("steer", {"angle": -0.2, "throttle": 0.1})],
        [("shoot", {"active": False, "power": -1.0})],
    ]
    batch = encoder.collate(enc_data)

    # 5. Field-token batch layout
    fig = plot_batch(encoder, batch)
    fig.savefig(OUT_DIR / "encoder_batch.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'encoder_batch.png'}")

    # 6. Token value grid
    fig = plot_token_grid(encoder, batch)
    fig.savefig(OUT_DIR / "encoder_token_grid.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'encoder_token_grid.png'}")

    # 7. Embedding norms
    enc_output = encoder(batch)
    fig = plot_embedding_norms(enc_output, batch)
    fig.savefig(OUT_DIR / "encoder_embedding_norms.png", **SAVEFIG_KW)
    print(f"  wrote {OUT_DIR / 'encoder_embedding_norms.png'}")

    print("Done.")


if __name__ == "__main__":
    main()
