"""Integration test: the Toy Game Controller from whitepaper Section 5.1."""

import torch
import pytest

from loomlib import (
    LoomModel,
    LoomUnion,
    LoomCompiler,
    Categorical,
    ContinuousScalar,
    BitInteger,
)


# -- Schema (matching the paper exactly) --

class MoveAction(LoomModel):
    direction: Categorical[4]           # N, S, E, W
    speed: ContinuousScalar[0.0, 10.0]  # bounded scalar

class CastSpell(LoomModel):
    spell_id: Categorical[6]            # spell selection
    power: BitInteger[8]                # 8-bit power (0-255)

class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell


class TestToyGameController:
    """Reproduces the walkthrough from Section 5 of the whitepaper."""

    @pytest.fixture
    def head(self):
        return LoomCompiler.build_head(AgentAction, d_model=256)

    def test_allocation_matches_paper(self, head):
        """Table from Section 4.2: opcode z[0:2], direction z[2:6],
        speed z[6:7], spell_id z[7:13], power z[13:21]."""
        alloc = head.allocation
        assert alloc.total_logits == 21
        assert alloc.opcode.start == 0 and alloc.opcode.end == 2
        entries = {e.name: (e.start, e.end) for e in alloc.all_entries()}
        assert entries["move.direction"] == (2, 6)
        assert entries["move.speed"] == (6, 7)
        assert entries["cast.spell_id"] == (7, 13)
        assert entries["cast.power"] == (13, 21)

    def test_forward_pass_decoding(self, head):
        """Section 5.1.1: given specific logits, decode Move(North, ~8.31)."""
        z = torch.zeros(1, 21)
        z[0, 0] = 2.5   # opcode: Move
        z[0, 1] = 0.1
        z[0, 2] = 0.9   # direction logits: North wins
        z[0, 3] = -0.9
        z[0, 4] = 0.1
        z[0, 5] = 0.3
        z[0, 6] = 0.8   # speed logit

        decoded = head.decode(z)
        assert decoded["__opcode__"].item() == 0  # Move
        assert decoded["move.direction"].item() == 0  # North
        speed = decoded["move.speed"].item()
        expected_speed = 5.0 + 5.0 * torch.tanh(torch.tensor(0.8)).item()
        assert speed == pytest.approx(expected_speed, abs=0.01)

    def test_loss_and_masking(self, head):
        """Section 5.1.2: loss for Move(North, 7.5) with spell params masked."""
        z = torch.randn(1, 21, requires_grad=True)
        targets = {
            "__opcode__": torch.tensor([0]),
            "direction": torch.tensor([0]),
            "speed": torch.tensor([7.5]),
        }
        z_masked = head.apply_gradient_mask(z, "move")
        total, breakdown = head.loss(z_masked, targets, active_branch="move")
        assert total.item() > 0
        total.backward()
        assert z.grad is not None

    def test_training_step(self, head):
        """Full training step: forward -> mask -> loss -> backward."""
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

        hidden = torch.randn(4, 256)
        targets = {
            "__opcode__": torch.tensor([0, 0, 1, 1]),
            "direction": torch.tensor([0, 1, 0, 0]),
            "speed": torch.tensor([5.0, 3.0, 0.0, 0.0]),
            "spell_id": torch.tensor([0, 0, 3, 5]),
            "power": torch.tensor([0, 0, 100, 200]),
        }

        # Forward
        z = head(hidden)

        # Compute losses for both branches and combine
        z_move = head.apply_gradient_mask(z, "move")
        loss_move, _ = head.loss(z_move, targets, active_branch="move")

        z_cast = head.apply_gradient_mask(z, "cast")
        loss_cast, _ = head.loss(z_cast, targets, active_branch="cast")

        loss = loss_move + loss_cast
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_encode_action_produces_embedding(self, head):
        """Re-entrant encoding maps decoded values back to d_model."""
        z = head(torch.randn(1, 256))
        decoded = head.decode(z)
        embedding = head.encode_action(decoded)
        assert embedding.shape[-1] == 256
