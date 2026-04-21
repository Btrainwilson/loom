"""Tests for loomlib.types -- encode/decode round-trips and loss gradients."""

import torch
import pytest

from loomlib.types import Categorical, ContinuousScalar, BitInteger, Boolean


class TestCategorical:
    def test_logit_size(self):
        c = Categorical(4)
        assert c.logit_size() == 4

    def test_class_getitem(self):
        c = Categorical[6]
        assert c.logit_size() == 6

    def test_decode_argmax(self):
        z = torch.tensor([0.1, 0.9, 0.0, -0.5])
        assert Categorical(4).decode(z).item() == 1

    def test_decode_batch(self):
        z = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = Categorical(3).decode(z)
        assert result.tolist() == [0, 2]

    def test_encode_onehot(self):
        c = Categorical(4)
        enc = c.encode(2)
        assert enc.shape == (4,)
        assert enc[2].item() == 1.0
        assert enc.sum().item() == 1.0

    def test_loss_cross_entropy(self):
        c = Categorical(3)
        z = torch.tensor([[2.0, 0.1, -1.0]], requires_grad=True)
        target = torch.tensor([0])
        loss = c.loss(z, target)
        assert loss.item() > 0
        loss.backward()
        assert z.grad is not None

    def test_round_trip(self):
        c = Categorical(5)
        for idx in range(5):
            enc = c.encode(idx)
            dec = c.decode(enc)
            assert dec.item() == idx


class TestContinuousScalar:
    def test_logit_size(self):
        assert ContinuousScalar(0.0, 10.0).logit_size() == 1

    def test_class_getitem(self):
        cs = ContinuousScalar[0.0, 10.0]
        assert cs._lo == 0.0
        assert cs._hi == 10.0

    def test_decode_range(self):
        cs = ContinuousScalar(0.0, 10.0)
        z_neg = torch.tensor([-100.0])
        z_pos = torch.tensor([100.0])
        assert cs.decode(z_neg).item() == pytest.approx(0.0, abs=0.01)
        assert cs.decode(z_pos).item() == pytest.approx(10.0, abs=0.01)

    def test_decode_midpoint(self):
        cs = ContinuousScalar(0.0, 10.0)
        z_zero = torch.tensor([0.0])
        assert cs.decode(z_zero).item() == pytest.approx(5.0, abs=0.01)

    def test_loss_mse(self):
        cs = ContinuousScalar(0.0, 10.0)
        z = torch.tensor([0.5], requires_grad=True)
        target = torch.tensor(7.0)
        loss = cs.loss(z, target)
        assert loss.item() > 0
        loss.backward()
        assert z.grad is not None

    def test_round_trip(self):
        cs = ContinuousScalar(-5.0, 5.0)
        for val in [-4.0, 0.0, 3.5]:
            enc = cs.encode(val)
            dec = cs.decode(enc)
            assert dec.item() == pytest.approx(val, abs=0.01)


class TestBitInteger:
    def test_logit_size(self):
        assert BitInteger(8).logit_size() == 8

    def test_class_getitem(self):
        b = BitInteger[8]
        assert b.logit_size() == 8

    def test_decode(self):
        bi = BitInteger(8)
        z = torch.tensor([10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
        # bits 0,1,2 = 1 -> 1+2+4 = 7
        assert bi.decode(z).item() == 7

    def test_encode(self):
        bi = BitInteger(8)
        enc = bi.encode(7)
        assert enc.shape == (8,)
        assert enc[:3].tolist() == [1.0, 1.0, 1.0]
        assert enc[3:].sum().item() == 0.0

    def test_loss_gradients(self):
        bi = BitInteger(8)
        z = torch.randn(8, requires_grad=True)
        target = torch.tensor(42)
        loss = bi.loss(z, target)
        loss.backward()
        assert z.grad is not None
        assert not torch.all(z.grad == 0)

    def test_round_trip(self):
        bi = BitInteger(8)
        for val in [0, 1, 127, 255]:
            enc = bi.encode(val)
            z = enc * 20.0 - 10.0  # push logits to extreme
            dec = bi.decode(z)
            assert dec.item() == val


class TestBoolean:
    def test_logit_size(self):
        assert Boolean().logit_size() == 1

    def test_decode(self):
        b = Boolean()
        assert b.decode(torch.tensor([5.0])).item() is True
        assert b.decode(torch.tensor([-5.0])).item() is False

    def test_encode(self):
        b = Boolean()
        enc_true = b.encode(True)
        enc_false = b.encode(False)
        assert enc_true.item() > 0
        assert enc_false.item() < 0

    def test_loss_gradients(self):
        b = Boolean()
        z = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor(1.0)
        loss = b.loss(z, target)
        loss.backward()
        assert z.grad is not None

    def test_round_trip(self):
        b = Boolean()
        for val in [True, False]:
            enc = b.encode(val)
            dec = b.decode(enc)
            assert dec.item() == val
