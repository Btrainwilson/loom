"""Tests for loomlib.slice -- SliceAllocation."""

import torch

from loomlib.slice import SliceAllocation, SliceEntry
from loomlib.types import Categorical, ContinuousScalar


class TestSliceAllocation:
    def test_build_gradient_mask(self):
        alloc = SliceAllocation()
        opcode = Categorical(2)
        alloc.add_opcode("__opcode__", 0, 2, opcode)
        alloc.add_branch("a")
        alloc.add_entry("a", "x", 2, 5, Categorical(3))
        alloc.add_branch("b")
        alloc.add_entry("b", "y", 5, 6, ContinuousScalar(0, 1))
        alloc.total_logits = 6

        mask_a = alloc.build_gradient_mask("a")
        assert mask_a.tolist() == [1, 1, 1, 1, 1, 0]

        mask_b = alloc.build_gradient_mask("b")
        assert mask_b.tolist() == [1, 1, 0, 0, 0, 1]

    def test_extract_slice(self):
        entry = SliceEntry("test", 2, 5, Categorical(3))
        z = torch.arange(10, dtype=torch.float)
        extracted = entry.extract(z)
        assert extracted.tolist() == [2.0, 3.0, 4.0]

    def test_extract_batched(self):
        entry = SliceEntry("test", 1, 3, Categorical(2))
        z = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float)
        extracted = entry.extract(z)
        assert extracted.shape == (2, 2)
        assert extracted[0].tolist() == [20.0, 30.0]
