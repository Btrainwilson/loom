"""Tests for loomlib.fn -- FnSpace and constraints."""

import torch
import pytest

from loomlib.fn import FnSpace, less_than_penalty, range_penalty


class TestFnSpace:
    def test_register_and_evaluate(self):
        fs = FnSpace()
        fs.add_fn("sum_xy", lambda x, y: x + y)

        values = {"x": torch.tensor(3.0), "y": torch.tensor(4.0)}
        result = fs.evaluate(values)
        assert result["sum_xy"].item() == 7.0

    def test_evaluation_order(self):
        fs = FnSpace()
        fs.add_fn("double_x", lambda x: x * 2)
        fs.add_fn("add_doubled", lambda double_x, y: double_x + y)

        values = {"x": torch.tensor(3.0), "y": torch.tensor(1.0)}
        result = fs.evaluate(values)
        assert result["double_x"].item() == 6.0
        assert result["add_doubled"].item() == 7.0

    def test_duplicate_fn_raises(self):
        fs = FnSpace()
        fs.add_fn("f", lambda x: x)
        with pytest.raises(ValueError, match="already registered"):
            fs.add_fn("f", lambda x: x + 1)


class TestConstraints:
    def test_less_than_satisfied(self):
        penalty = less_than_penalty(torch.tensor(1.0), torch.tensor(5.0))
        assert penalty.item() == 0.0

    def test_less_than_violated(self):
        penalty = less_than_penalty(torch.tensor(5.0), torch.tensor(1.0))
        assert penalty.item() == pytest.approx(4.0)

    def test_range_penalty_in_range(self):
        penalty = range_penalty(torch.tensor(5.0), 0.0, 10.0)
        assert penalty.item() == 0.0

    def test_range_penalty_below(self):
        penalty = range_penalty(torch.tensor(-2.0), 0.0, 10.0)
        assert penalty.item() == pytest.approx(2.0)

    def test_range_penalty_above(self):
        penalty = range_penalty(torch.tensor(12.0), 0.0, 10.0)
        assert penalty.item() == pytest.approx(2.0)
