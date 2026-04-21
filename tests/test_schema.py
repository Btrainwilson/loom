"""Tests for loomlib.schema -- LoomModel and LoomUnion annotation introspection."""

import pytest

from loomlib import LoomModel, LoomUnion, Categorical, ContinuousScalar, BitInteger


class TestLoomModel:
    def test_fields_collected(self):
        class MyModel(LoomModel):
            direction: Categorical[4]
            speed: ContinuousScalar[0.0, 10.0]

        fields = MyModel.loom_fields()
        assert list(fields.keys()) == ["direction", "speed"]
        assert isinstance(fields["direction"], Categorical)
        assert isinstance(fields["speed"], ContinuousScalar)

    def test_total_logits(self):
        class MyModel(LoomModel):
            direction: Categorical[4]
            speed: ContinuousScalar[0.0, 10.0]
            power: BitInteger[8]

        assert MyModel.total_logits() == 4 + 1 + 8

    def test_private_fields_ignored(self):
        class MyModel(LoomModel):
            _hidden: Categorical[2]
            visible: Categorical[3]

        # _hidden should be skipped
        assert list(MyModel.loom_fields().keys()) == ["visible"]

    def test_invalid_annotation_raises(self):
        with pytest.raises(TypeError, match="Cannot resolve"):
            class BadModel(LoomModel):
                x: str  # not a LoomType or supported native type


class TestLoomUnion:
    def test_branches_collected(self):
        class A(LoomModel):
            x: Categorical[2]

        class B(LoomModel):
            y: ContinuousScalar[0.0, 1.0]

        class MyUnion(LoomUnion):
            branch_a: A
            branch_b: B

        branches = MyUnion.loom_branches()
        assert list(branches.keys()) == ["branch_a", "branch_b"]
        assert branches["branch_a"] is A
        assert branches["branch_b"] is B

    def test_num_branches(self):
        class A(LoomModel):
            x: Categorical[2]

        class B(LoomModel):
            y: Categorical[3]

        class U(LoomUnion):
            a: A
            b: B

        assert U.num_branches() == 2

    def test_non_loommodel_branch_raises(self):
        with pytest.raises(TypeError, match="must be a LoomModel subclass"):
            class BadUnion(LoomUnion):
                a: int
