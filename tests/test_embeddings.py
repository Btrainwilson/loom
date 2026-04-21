"""Tests for branch embedding modules: XVal, Thermometer, Fourier, GaussianBasis."""

import torch
import torch.nn as nn
import pytest

from loomlib import (
    LoomModel,
    LoomCompiler,
    LoomEncoder,
    Scalar,
    ContinuousScalar,
    XValEmbedding,
    ThermometerEmbedding,
    FourierValueEmbedding,
    GaussianBasisEmbedding,
)


# -- Schemas for integration tests -------------------------------------------

class TwoScalars(LoomModel):
    x: Scalar()
    y: Scalar()


D_MODEL = 32
NUM_FIELDS = 2
N_TOKENS = 5


def _sample_inputs():
    field_ids = torch.tensor([0, 1, 0, 1, 0])
    values = torch.tensor([1.0, -2.5, 0.0, 100.0, 0.01])
    return field_ids, values


# ======================================================================
# XValEmbedding
# ======================================================================

class TestXValEmbedding:
    def test_output_shape(self):
        emb = XValEmbedding(NUM_FIELDS, D_MODEL, k=2)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        assert out.shape == (N_TOKENS, D_MODEL)

    def test_k_zero_reduces_to_single_scale(self):
        emb = XValEmbedding(NUM_FIELDS, D_MODEL, k=0)
        assert emb.num_scales == 1
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        assert out.shape == (N_TOKENS, D_MODEL)

    def test_scale_factors_shape(self):
        emb = XValEmbedding(NUM_FIELDS, D_MODEL, k=3)
        assert emb.scale_factors.shape == (7,)
        assert emb.scale_factors[3].item() == pytest.approx(1.0)  # 10^0

    def test_gradient_flow(self):
        emb = XValEmbedding(NUM_FIELDS, D_MODEL, k=1)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        out.sum().backward()
        for name, p in emb.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_values_different_output(self):
        emb = XValEmbedding(1, D_MODEL, k=2)
        ids = torch.tensor([0, 0])
        vals = torch.tensor([0.0, 10.0])
        out = emb(ids, vals)
        assert not torch.allclose(out[0], out[1])


# ======================================================================
# ThermometerEmbedding
# ======================================================================

class TestThermometerEmbedding:
    def test_output_shape(self):
        emb = ThermometerEmbedding(NUM_FIELDS, D_MODEL, num_buckets=32)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        assert out.shape == (N_TOKENS, D_MODEL)

    def test_thresholds_count(self):
        emb = ThermometerEmbedding(NUM_FIELDS, D_MODEL, num_buckets=16, val_min=0, val_max=100)
        assert emb.thresholds.shape == (16,)
        assert emb.thresholds[0].item() == pytest.approx(0.0)
        assert emb.thresholds[-1].item() == pytest.approx(100.0)

    def test_gradient_flow(self):
        emb = ThermometerEmbedding(NUM_FIELDS, D_MODEL, num_buckets=16)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        out.sum().backward()
        for name, p in emb.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_below_min_all_zeros(self):
        emb = ThermometerEmbedding(1, D_MODEL, num_buckets=8, val_min=10.0, val_max=80.0)
        ids = torch.tensor([0])
        vals = torch.tensor([5.0])
        # Manually check the thermo vector
        thermo = (vals.unsqueeze(-1) >= emb.thresholds).float()
        assert thermo.sum().item() == 0.0


# ======================================================================
# FourierValueEmbedding
# ======================================================================

class TestFourierValueEmbedding:
    def test_output_shape(self):
        emb = FourierValueEmbedding(NUM_FIELDS, D_MODEL, num_frequencies=16)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        assert out.shape == (N_TOKENS, D_MODEL)

    def test_frequencies_shape(self):
        emb = FourierValueEmbedding(NUM_FIELDS, D_MODEL, num_frequencies=24)
        assert emb.freqs.shape == (24,)

    def test_gradient_flow(self):
        emb = FourierValueEmbedding(NUM_FIELDS, D_MODEL, num_frequencies=16)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        out.sum().backward()
        for name, p in emb.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_zero_value_still_has_cos_component(self):
        """cos(0) = 1, so the raw features aren't all zero at v=0."""
        emb = FourierValueEmbedding(1, D_MODEL, num_frequencies=8)
        ids = torch.tensor([0])
        vals = torch.tensor([0.0])
        out = emb(ids, vals)
        assert out.norm().item() > 0.0


# ======================================================================
# GaussianBasisEmbedding
# ======================================================================

class TestGaussianBasisEmbedding:
    def test_output_shape(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=32)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        assert out.shape == (N_TOKENS, D_MODEL)

    def test_centers_range(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=10, val_min=-5.0, val_max=5.0)
        assert emb.centers[0].item() == pytest.approx(-5.0)
        assert emb.centers[-1].item() == pytest.approx(5.0)

    def test_learnable_centers(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=8, learnable_centers=True)
        assert isinstance(emb.centers, nn.Parameter)
        assert isinstance(emb.sigma, nn.Parameter)

    def test_fixed_centers(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=8, learnable_centers=False)
        assert not isinstance(emb.centers, nn.Parameter)

    def test_gradient_flow(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=16)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        out.sum().backward()
        for name, p in emb.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_learnable_centers_gradient(self):
        emb = GaussianBasisEmbedding(NUM_FIELDS, D_MODEL, num_centers=8, learnable_centers=True)
        field_ids, values = _sample_inputs()
        out = emb(field_ids, values)
        out.sum().backward()
        assert emb.centers.grad is not None
        assert emb.sigma.grad is not None

    def test_peak_at_center(self):
        """Value exactly at a center should activate that basis strongly."""
        emb = GaussianBasisEmbedding(1, D_MODEL, num_centers=5, val_min=0.0, val_max=4.0)
        ids = torch.tensor([0])
        vals = torch.tensor([2.0])  # center index 2 = value 2.0
        diff = vals.unsqueeze(-1) - emb.centers
        phi = torch.exp(-0.5 * (diff / emb.sigma) ** 2)
        assert phi[0, 2].item() == pytest.approx(1.0, abs=1e-5)


# ======================================================================
# LoomEncoder integration
# ======================================================================

class TestEncoderIntegration:
    """Each embedding module works as a drop-in branch_embeddings override."""

    def _build_and_run(self, branch_emb: nn.Module):
        encoder = LoomCompiler.build_encoder(
            TwoScalars,
            d_model=D_MODEL,
            branch_embeddings={"__root__": branch_emb},
        )
        data = [[("__root__", {"x": 1.5, "y": -0.3})]]
        batch = encoder.collate(data)
        out = encoder(batch)
        assert out.shape == (1, 2, D_MODEL)
        # Gradient flow through entire encoder
        out.sum().backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_xval_integration(self):
        self._build_and_run(XValEmbedding(2, D_MODEL, k=1))

    def test_thermometer_integration(self):
        self._build_and_run(ThermometerEmbedding(2, D_MODEL, num_buckets=16))

    def test_fourier_integration(self):
        self._build_and_run(FourierValueEmbedding(2, D_MODEL, num_frequencies=8))

    def test_gaussian_integration(self):
        self._build_and_run(GaussianBasisEmbedding(2, D_MODEL, num_centers=16))


# ======================================================================
# Encoder -> Transformer -> Head round-trip
# ======================================================================

class TestRoundTrip:
    def test_xval_end_to_end(self):
        encoder = LoomCompiler.build_encoder(
            TwoScalars,
            d_model=D_MODEL,
            branch_embeddings={"__root__": XValEmbedding(2, D_MODEL, k=1)},
        )
        head = LoomCompiler.build_head(TwoScalars, d_model=D_MODEL)

        data = [[("__root__", {"x": 3.0, "y": -1.0})]]
        batch = encoder.collate(data)
        emb = encoder(batch)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=64, batch_first=True,
        )
        transformer = nn.TransformerEncoder(layer, num_layers=1)
        hidden = transformer(emb, src_key_padding_mask=batch.padding_mask)
        pooled = hidden.mean(dim=1)

        z = head(pooled)
        decoded = head.decode(z)
        assert "x" in decoded
        assert "y" in decoded

        # Gradients propagate end-to-end
        z.sum().backward()
        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert enc_has_grad
