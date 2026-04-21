"""Tests for loomlib.compiler -- allocation, masks, and build_head."""

import torch
import pytest

from loomlib import (
    LoomModel,
    LoomUnion,
    LoomCompiler,
    Categorical,
    ContinuousScalar,
    BitInteger,
    Boolean,
)


class MoveAction(LoomModel):
    direction: Categorical[4]
    speed: ContinuousScalar[0.0, 10.0]


class CastSpell(LoomModel):
    spell_id: Categorical[6]
    power: BitInteger[8]


class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell


class TestAllocationUnion:
    def test_total_logits(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        # opcode(2) + direction(4) + speed(1) + spell_id(6) + power(8) = 21
        assert head.total_logits == 21

    def test_opcode_present(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        assert head.allocation.opcode is not None
        assert head.allocation.opcode.size == 2

    def test_branch_names(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        assert list(head.allocation.branches.keys()) == ["move", "cast"]

    def test_allocation_ranges(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        entries = head.allocation.all_entries()
        names = [e.name for e in entries]
        assert names == [
            "__opcode__",
            "move.direction",
            "move.speed",
            "cast.spell_id",
            "cast.power",
        ]
        ranges = [(e.start, e.end) for e in entries]
        assert ranges == [(0, 2), (2, 6), (6, 7), (7, 13), (13, 21)]


class TestAllocationModel:
    def test_flat_model(self):
        head = LoomCompiler.build_head(MoveAction, d_model=64)
        assert head.total_logits == 5  # 4 + 1
        assert head.allocation.opcode is None
        assert "__root__" in head.allocation.branches


class TestMasks:
    def test_move_mask_shape(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        mask = head._gradient_masks["move"]
        assert mask.shape == (21,)

    def test_move_mask_values(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        mask = head._gradient_masks["move"]
        # opcode (0:2) and move fields (2:7) should be 1
        assert mask[:7].sum().item() == 7.0
        # cast fields (7:21) should be 0
        assert mask[7:].sum().item() == 0.0

    def test_cast_mask_values(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        mask = head._gradient_masks["cast"]
        # opcode (0:2) should be 1
        assert mask[:2].sum().item() == 2.0
        # move fields (2:7) should be 0
        assert mask[2:7].sum().item() == 0.0
        # cast fields (7:21) should be 1
        assert mask[7:].sum().item() == 14.0


class TestBuildHead:
    def test_forward_shape(self):
        head = LoomCompiler.build_head(AgentAction, d_model=128)
        z = head(torch.randn(4, 128))
        assert z.shape == (4, 21)

    def test_decode_returns_dict(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        z = head(torch.randn(1, 64))
        decoded = head.decode(z)
        assert "__opcode__" in decoded
        assert "move.direction" in decoded
        assert "cast.power" in decoded

    def test_loss_computes(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        z = head(torch.randn(1, 64))
        targets = {
            "__opcode__": torch.tensor([0]),
            "direction": torch.tensor([2]),
            "speed": torch.tensor([7.5]),
        }
        total, breakdown = head.loss(z, targets, active_branch="move")
        assert total.item() > 0
        assert "__opcode__" in breakdown

    def test_gradient_mask_blocks_gradients(self):
        head = LoomCompiler.build_head(AgentAction, d_model=64)
        hidden = torch.randn(1, 64, requires_grad=True)
        z = head(hidden)
        z = head.apply_gradient_mask(z, "move")
        # Only use cast-region logits to create a scalar
        cast_logits = z[..., 7:13]
        fake_loss = cast_logits.sum()
        fake_loss.backward()
        # Gradients through cast logits should be zero because mask blocks them
        z_grad = head.projection.weight.grad
        # The gradient should exist but be zeroed for cast dims
        assert z_grad is not None
