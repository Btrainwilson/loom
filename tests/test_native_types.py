"""Tests for native Python type support in LoomModel schemas."""

import torch
import pytest

from loomlib import (
    LoomModel,
    LoomUnion,
    LoomCompiler,
    Categorical,
    Scalar,
)
from loomlib.types import Boolean, BitInteger


# ---------------------------------------------------------------------------
# Scalar type unit tests
# ---------------------------------------------------------------------------

class TestScalar:
    def test_logit_size(self):
        assert Scalar().logit_size() == 1

    def test_decode_identity(self):
        s = Scalar()
        z = torch.tensor([3.14])
        assert s.decode(z).item() == pytest.approx(3.14)

    def test_decode_unbounded(self):
        s = Scalar()
        assert s.decode(torch.tensor([1000.0])).item() == pytest.approx(1000.0)
        assert s.decode(torch.tensor([-1000.0])).item() == pytest.approx(-1000.0)

    def test_encode(self):
        s = Scalar()
        enc = s.encode(42.0)
        assert enc.shape == (1,)
        assert enc.item() == pytest.approx(42.0)

    def test_round_trip(self):
        s = Scalar()
        for val in [-100.0, 0.0, 0.001, 999.9]:
            enc = s.encode(val)
            dec = s.decode(enc)
            assert dec.item() == pytest.approx(val)

    def test_loss_gradient(self):
        s = Scalar()
        z = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor(5.0)
        loss = s.loss(z, target)
        assert loss.item() > 0
        loss.backward()
        assert z.grad is not None
        assert z.grad.item() < 0  # gradient pushes z toward 5


# ---------------------------------------------------------------------------
# Native type resolution in LoomModel
# ---------------------------------------------------------------------------

class TestNativeTypeResolution:
    def test_bool_resolves_to_boolean(self):
        class M(LoomModel):
            flag: bool

        thunk = M.loom_fields()["flag"]
        assert isinstance(thunk, Boolean)
        assert thunk.logit_size() == 1

    def test_int_resolves_to_bit_integer_32(self):
        class M(LoomModel):
            count: int

        thunk = M.loom_fields()["count"]
        assert isinstance(thunk, BitInteger)
        assert thunk.logit_size() == 32

    def test_float_resolves_to_scalar(self):
        class M(LoomModel):
            velocity: float

        thunk = M.loom_fields()["velocity"]
        assert isinstance(thunk, Scalar)
        assert thunk.logit_size() == 1

    def test_total_logits_native(self):
        class M(LoomModel):
            alive: bool       # 1
            health: int       # 32
            speed: float      # 1

        assert M.total_logits() == 34


# ---------------------------------------------------------------------------
# Mixed schemas (native + explicit Loom types)
# ---------------------------------------------------------------------------

class TestMixedSchemas:
    def test_mixed_model_fields(self):
        class M(LoomModel):
            direction: Categorical[4]
            alive: bool
            speed: float

        fields = M.loom_fields()
        assert list(fields.keys()) == ["direction", "alive", "speed"]
        assert isinstance(fields["direction"], Categorical)
        assert isinstance(fields["alive"], Boolean)
        assert isinstance(fields["speed"], Scalar)
        assert M.total_logits() == 4 + 1 + 1

    def test_mixed_compile_and_forward(self):
        class M(LoomModel):
            direction: Categorical[4]
            alive: bool
            speed: float

        head = LoomCompiler.build_head(M, d_model=64)
        assert head.total_logits == 6

        z = head(torch.randn(2, 64))
        assert z.shape == (2, 6)

        decoded = head.decode(z)
        assert "direction" in decoded
        assert "alive" in decoded
        assert "speed" in decoded

    def test_native_in_union(self):
        class Attack(LoomModel):
            target_id: Categorical[8]
            power: float

        class Heal(LoomModel):
            amount: int

        class Action(LoomUnion):
            attack: Attack
            heal: Heal

        head = LoomCompiler.build_head(Action, d_model=64)
        # opcode(2) + target_id(8) + power(1) + amount(32) = 43
        assert head.total_logits == 43

        z = head(torch.randn(1, 64))
        decoded = head.decode(z)
        assert "__opcode__" in decoded

    def test_loss_with_native_types(self):
        class M(LoomModel):
            flag: bool
            value: float

        head = LoomCompiler.build_head(M, d_model=32)
        z = head(torch.randn(1, 32))
        targets = {
            "flag": torch.tensor([1.0]),
            "value": torch.tensor([3.5]),
        }
        total, breakdown = head.loss(z, targets)
        assert total.item() > 0
        assert "flag" in breakdown
        assert "value" in breakdown

    def test_int_round_trip_via_head(self):
        class M(LoomModel):
            count: int

        head = LoomCompiler.build_head(M, d_model=32)
        # Encode a known value -> feed as logits -> decode
        thunk = M.loom_fields()["count"]
        encoded = thunk.encode(42)
        z_strong = encoded * 20.0 - 10.0  # push logits to extremes
        decoded_val = thunk.decode(z_strong)
        assert decoded_val.item() == 42
