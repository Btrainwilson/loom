"""Tests for LoomEncoder: collation, forward, custom embeddings, round-trip."""

import torch
import torch.nn as nn
import pytest

from loomlib import (
    LoomModel,
    LoomUnion,
    LoomCompiler,
    LoomEncoder,
    LoomBatch,
    DefaultBranchEmbedding,
    Boolean,
    Scalar,
    ContinuousScalar,
)


# -- Schemas using scalar-encoding types ------------------------------------

class Steering(LoomModel):
    angle: ContinuousScalar[-1.0, 1.0]
    throttle: ContinuousScalar[0.0, 1.0]

class Shooting(LoomModel):
    active: Boolean()
    power: Scalar()

class GameAction(LoomUnion):
    steer: Steering
    shoot: Shooting


class FlatMove(LoomModel):
    x: Scalar()
    y: Scalar()


# -- Fixtures ---------------------------------------------------------------

D_MODEL = 32


@pytest.fixture
def encoder():
    return LoomCompiler.build_encoder(GameAction, d_model=D_MODEL)


@pytest.fixture
def flat_encoder():
    return LoomCompiler.build_encoder(FlatMove, d_model=D_MODEL)


def _steer_inst(angle: float, throttle: float) -> tuple[str, dict]:
    return ("steer", {"angle": angle, "throttle": throttle})


def _shoot_inst(active: bool, power: float) -> tuple[str, dict]:
    return ("shoot", {"active": active, "power": power})


def _sample_batch() -> list[list[tuple[str, dict]]]:
    return [
        [_steer_inst(0.5, 0.8), _shoot_inst(True, 3.0)],
        [_steer_inst(-0.2, 0.1)],
    ]


# -- Collation shapes -------------------------------------------------------

class TestCollationShapes:

    def test_tensor_shapes(self, encoder):
        batch = encoder.collate(_sample_batch())
        # seq 0: 2 fields (steer) + 2 fields (shoot) = 4
        # seq 1: 2 fields (steer) = 2  ->  N_max = 4
        assert batch.type_ids.shape == (2, 4)
        assert batch.inst_ids.shape == (2, 4)
        assert batch.field_ids.shape == (2, 4)
        assert batch.values.shape == (2, 4)
        assert batch.padding_mask.shape == (2, 4)

    def test_dtypes(self, encoder):
        batch = encoder.collate(_sample_batch())
        assert batch.type_ids.dtype == torch.long
        assert batch.inst_ids.dtype == torch.long
        assert batch.field_ids.dtype == torch.long
        assert batch.values.dtype == torch.float32
        assert batch.padding_mask.dtype == torch.bool

    def test_padding_counts(self, encoder):
        batch = encoder.collate(_sample_batch())
        # seq 0: 4 real, 0 pad; seq 1: 2 real, 2 pad
        assert batch.padding_mask[0].sum().item() == 0
        assert batch.padding_mask[1].sum().item() == 2

    def test_batch_properties(self, encoder):
        batch = encoder.collate(_sample_batch())
        assert batch.batch_size == 2
        assert batch.seq_len == 4


# -- Collation values -------------------------------------------------------

class TestCollationValues:

    def test_type_ids(self, encoder):
        batch = encoder.collate(_sample_batch())
        steer_idx = encoder.branch_idx["steer"]
        shoot_idx = encoder.branch_idx["shoot"]
        assert batch.type_ids[0, 0].item() == steer_idx
        assert batch.type_ids[0, 1].item() == steer_idx
        assert batch.type_ids[0, 2].item() == shoot_idx
        assert batch.type_ids[0, 3].item() == shoot_idx

    def test_inst_ids(self, encoder):
        batch = encoder.collate(_sample_batch())
        # seq 0: inst 0 (steer, 2 fields), inst 1 (shoot, 2 fields)
        assert batch.inst_ids[0, 0].item() == 0
        assert batch.inst_ids[0, 1].item() == 0
        assert batch.inst_ids[0, 2].item() == 1
        assert batch.inst_ids[0, 3].item() == 1

    def test_encoded_values(self, encoder):
        """Values tensor should match LoomType.encode().item() for each field."""
        batch = encoder.collate(_sample_batch())
        angle_thunk = encoder.branch_fields["steer"]["angle"]
        throttle_thunk = encoder.branch_fields["steer"]["throttle"]

        expected_angle = angle_thunk.encode(0.5).item()
        expected_throttle = throttle_thunk.encode(0.8).item()
        assert batch.values[0, 0].item() == pytest.approx(expected_angle, abs=1e-5)
        assert batch.values[0, 1].item() == pytest.approx(expected_throttle, abs=1e-5)


# -- Global field ids -------------------------------------------------------

class TestGlobalFieldIds:

    def test_field_id_offsets(self, encoder):
        """Branch 0 fields start at 0; branch 1 fields are offset."""
        steer_offset = encoder._field_offsets["steer"]
        shoot_offset = encoder._field_offsets["shoot"]
        num_steer_fields = len(encoder.branch_fields["steer"])
        assert steer_offset == 0
        assert shoot_offset == num_steer_fields

    def test_field_ids_in_batch(self, encoder):
        batch = encoder.collate(_sample_batch())
        steer_ids = [
            encoder._global_field_id["steer"][f]
            for f in encoder.branch_fields["steer"]
        ]
        shoot_ids = [
            encoder._global_field_id["shoot"][f]
            for f in encoder.branch_fields["shoot"]
        ]
        assert batch.field_ids[0, 0].item() == steer_ids[0]
        assert batch.field_ids[0, 1].item() == steer_ids[1]
        assert batch.field_ids[0, 2].item() == shoot_ids[0]
        assert batch.field_ids[0, 3].item() == shoot_ids[1]


# -- Forward shape and masking ----------------------------------------------

class TestForward:

    def test_output_shape(self, encoder):
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)
        assert out.shape == (2, 4, D_MODEL)

    def test_padding_zeros(self, encoder):
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)
        # seq 1 has padding at positions 2, 3
        assert out[1, 2].norm().item() == 0.0
        assert out[1, 3].norm().item() == 0.0

    def test_real_tokens_nonzero(self, encoder):
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)
        assert out[0, 0].norm().item() > 0.0
        assert out[1, 0].norm().item() > 0.0


# -- Flat model (single __root__ branch) -----------------------------------

class TestFlatModel:

    def test_flat_encoder_branches(self, flat_encoder):
        assert flat_encoder.branch_names == ["__root__"]

    def test_flat_collation(self, flat_encoder):
        data = [[("__root__", {"x": 1.5, "y": -0.3})]]
        batch = flat_encoder.collate(data)
        assert batch.type_ids.shape == (1, 2)
        assert batch.padding_mask.sum().item() == 0

    def test_flat_forward(self, flat_encoder):
        data = [[("__root__", {"x": 1.5, "y": -0.3})]]
        batch = flat_encoder.collate(data)
        out = flat_encoder(batch)
        assert out.shape == (1, 2, D_MODEL)


# -- Custom branch embedding -----------------------------------------------

class TestCustomBranchEmbedding:

    def test_custom_module_is_used(self):
        call_log = []

        class TrackedEmbedding(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.proj = nn.Linear(1, d_model)

            def forward(self, field_ids, values):
                call_log.append(field_ids.shape[0])
                return self.proj(values.unsqueeze(-1))

        custom = {"shoot": TrackedEmbedding(D_MODEL)}
        encoder = LoomCompiler.build_encoder(
            GameAction, d_model=D_MODEL, branch_embeddings=custom,
        )
        batch = encoder.collate(_sample_batch())
        encoder(batch)

        assert len(call_log) > 0, "Custom embedding was never called"
        assert isinstance(encoder.branch_encoders["shoot"], TrackedEmbedding)
        assert isinstance(encoder.branch_encoders["steer"], DefaultBranchEmbedding)

    def test_custom_output_in_correct_positions(self):
        class ConstantEmbedding(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model

            def forward(self, field_ids, values):
                return torch.ones(field_ids.shape[0], self.d_model) * 999.0

        encoder = LoomCompiler.build_encoder(
            GameAction, d_model=D_MODEL,
            branch_embeddings={"shoot": ConstantEmbedding(D_MODEL)},
        )
        data = [[_shoot_inst(True, 1.0)]]
        batch = encoder.collate(data)
        out = encoder(batch)
        # The branch_emb contribution is 999.0 per dim, plus the type+inst embedding
        # so output should have values substantially > 0
        assert out[0, 0].mean().item() > 10.0


# -- Batch.to() -------------------------------------------------------------

class TestBatchTo:

    def test_to_preserves_data(self, encoder):
        batch = encoder.collate(_sample_batch())
        moved = batch.to("cpu")
        assert torch.equal(batch.values, moved.values)
        assert torch.equal(batch.padding_mask, moved.padding_mask)


# -- Gradient flow ----------------------------------------------------------

class TestGradientFlow:

    def test_gradients_reach_all_parameters(self, encoder):
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_reach_custom_module(self):
        custom_mod = nn.Linear(1, D_MODEL)
        branch_emb = _WrapLinear(custom_mod, D_MODEL)
        encoder = LoomCompiler.build_encoder(
            GameAction, d_model=D_MODEL,
            branch_embeddings={"shoot": branch_emb},
        )
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)
        out.sum().backward()

        assert custom_mod.weight.grad is not None


class _WrapLinear(nn.Module):
    def __init__(self, linear: nn.Linear, d_model: int):
        super().__init__()
        self.linear = linear
        self.d_model = d_model

    def forward(self, field_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values.unsqueeze(-1))


# -- padding_mask convention -------------------------------------------------

class TestPaddingMaskConvention:

    def test_works_as_src_key_padding_mask(self, encoder):
        """padding_mask can be passed directly to nn.TransformerEncoder."""
        batch = encoder.collate(_sample_batch())
        out = encoder(batch)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=64, batch_first=True,
        )
        transformer = nn.TransformerEncoder(layer, num_layers=1)
        result = transformer(out, src_key_padding_mask=batch.padding_mask)
        assert result.shape == out.shape


# -- Round-trip: encoder -> transformer -> LoomHead.decode ------------------

class TestRoundTrip:

    def test_encoder_to_head(self):
        encoder = LoomCompiler.build_encoder(GameAction, d_model=D_MODEL)
        head = LoomCompiler.build_head(GameAction, d_model=D_MODEL)

        data = [[_steer_inst(0.3, 0.7), _shoot_inst(False, -1.0)]]
        batch = encoder.collate(data)
        emb = encoder(batch)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=64, batch_first=True,
        )
        transformer = nn.TransformerEncoder(layer, num_layers=1)
        hidden = transformer(emb, src_key_padding_mask=batch.padding_mask)

        # Pool over sequence -> single vector per batch
        real = ~batch.padding_mask
        pooled = (hidden * real.unsqueeze(-1).float()).sum(dim=1) / real.sum(dim=1, keepdim=True).float()

        z = head(pooled)
        decoded = head.decode(z)

        assert "__opcode__" in decoded
        assert "steer.angle" in decoded
        assert "shoot.power" in decoded

    def test_gradients_flow_end_to_end(self):
        encoder = LoomCompiler.build_encoder(GameAction, d_model=D_MODEL)
        head = LoomCompiler.build_head(GameAction, d_model=D_MODEL)

        data = [[_steer_inst(0.3, 0.7)]]
        batch = encoder.collate(data)
        emb = encoder(batch)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=64, batch_first=True,
        )
        transformer = nn.TransformerEncoder(layer, num_layers=1)
        hidden = transformer(emb, src_key_padding_mask=batch.padding_mask)
        pooled = hidden.mean(dim=1)

        z = head(pooled)
        loss = z.sum()
        loss.backward()

        # Encoder params get gradients
        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert enc_has_grad, "Encoder received no gradients"

        # Head params get gradients
        head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert head_has_grad, "Head received no gradients"
