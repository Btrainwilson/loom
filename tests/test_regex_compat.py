"""Tests for the regex -> LoomModel/LoomUnion converter."""

import math

import pytest
import torch

from loomlib import (
    LoomCompiler,
    LoomModel,
    LoomUnion,
    Categorical,
    BitInteger,
    Boolean,
    Scalar,
)
from loomlib.compat.regex import from_regex, RegexSchema


# ======================================================================
# Flat model — literal alternation
# ======================================================================

class TestLiteralAlternation:
    def test_basic_alternation(self):
        rs = from_regex(r"(?P<level>DEBUG|INFO|WARN|ERROR)")
        assert issubclass(rs.schema, LoomModel)
        fields = rs.schema.loom_fields()
        assert list(fields.keys()) == ["level"]
        assert isinstance(fields["level"], Categorical)
        assert fields["level"].logit_size() == 4

    def test_vocabulary_order(self):
        rs = from_regex(r"(?P<color>red|green|blue)")
        assert rs.vocabularies == {"color": ["red", "green", "blue"]}

    def test_custom_name(self):
        rs = from_regex(r"(?P<x>a|b)", name="MyModel")
        assert rs.schema.__name__ == "MyModel"

    def test_default_name(self):
        rs = from_regex(r"(?P<x>a|b)")
        assert rs.schema.__name__ == "RegexModel"


# ======================================================================
# Flat model — character class
# ======================================================================

class TestCharacterClass:
    def test_explicit_chars(self):
        rs = from_regex(r"(?P<dir>[NSEW])")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["dir"], Categorical)
        assert fields["dir"].logit_size() == 4
        assert rs.vocabularies["dir"] == ["N", "S", "E", "W"]

    def test_char_range(self):
        rs = from_regex(r"(?P<grade>[A-F])")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["grade"], Categorical)
        assert fields["grade"].logit_size() == 6
        assert rs.vocabularies["grade"] == ["A", "B", "C", "D", "E", "F"]

    def test_single_digit_class(self):
        rs = from_regex(r"(?P<d>[0-9])")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["d"], Categorical)
        assert fields["d"].logit_size() == 10

    def test_digit_shorthand(self):
        rs = from_regex(r"(?P<d>\d)")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["d"], Categorical)
        assert fields["d"].logit_size() == 10


# ======================================================================
# Flat model — digit repetition (BitInteger)
# ======================================================================

class TestDigitRepetition:
    def test_fixed_digits(self):
        rs = from_regex(r"(?P<code>\d{3})")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["code"], BitInteger)
        expected_bits = math.ceil(math.log2(1000))
        assert fields["code"].logit_size() == expected_bits

    def test_unbounded_digits(self):
        rs = from_regex(r"(?P<num>\d+)")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["num"], BitInteger)
        assert fields["num"].logit_size() == 32

    def test_bounded_range_digits(self):
        rs = from_regex(r"(?P<port>\d{1,5})")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["port"], BitInteger)
        max_val = 10**5 - 1  # 99999
        expected_bits = math.ceil(math.log2(max_val + 1))
        assert fields["port"].logit_size() == expected_bits

    def test_no_vocabulary_for_integers(self):
        rs = from_regex(r"(?P<n>\d{4})")
        assert "n" not in rs.vocabularies


# ======================================================================
# Flat model — boolean
# ======================================================================

class TestBoolean:
    def test_true_false(self):
        rs = from_regex(r"(?P<flag>true|false)")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["flag"], Boolean)
        assert "flag" not in rs.vocabularies

    def test_zero_one(self):
        rs = from_regex(r"(?P<bit>0|1)")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["bit"], Boolean)

    def test_yes_no(self):
        rs = from_regex(r"(?P<ok>yes|no)")
        # "yes"/"no" are not in _BOOL_TRUE/_BOOL_FALSE sets when both
        # mixed-case variants aren't present — should be Categorical
        # unless both are in the boolean sets
        fields = rs.schema.loom_fields()
        assert isinstance(fields["ok"], Boolean)


# ======================================================================
# Flat model — float (Scalar)
# ======================================================================

class TestFloat:
    def test_float_pattern(self):
        rs = from_regex(r"(?P<val>\d+\.\d+)")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["val"], Scalar)
        assert fields["val"].logit_size() == 1

    def test_fixed_width_float(self):
        rs = from_regex(r"(?P<temp>\d{1,3}\.\d{2})")
        fields = rs.schema.loom_fields()
        assert isinstance(fields["temp"], Scalar)


# ======================================================================
# Flat model — multiple fields
# ======================================================================

class TestMultipleFields:
    def test_log_entry(self):
        rs = from_regex(
            r"(?P<level>DEBUG|INFO|WARN|ERROR) (?P<code>\d{3}) (?P<latency>\d+\.\d+)",
            name="LogEntry",
        )
        fields = rs.schema.loom_fields()
        assert list(fields.keys()) == ["level", "code", "latency"]
        assert isinstance(fields["level"], Categorical)
        assert isinstance(fields["code"], BitInteger)
        assert isinstance(fields["latency"], Scalar)
        assert rs.schema.__name__ == "LogEntry"

    def test_field_order_preserved(self):
        rs = from_regex(r"(?P<a>x|y) (?P<b>\d+) (?P<c>true|false)")
        assert list(rs.schema.loom_fields().keys()) == ["a", "b", "c"]

    def test_literals_between_groups_ignored(self):
        rs = from_regex(r"START:(?P<x>a|b):MIDDLE:(?P<y>\d+):END")
        fields = rs.schema.loom_fields()
        assert list(fields.keys()) == ["x", "y"]


# ======================================================================
# Union — top-level alternation
# ======================================================================

class TestUnion:
    def test_basic_union(self):
        rs = from_regex(
            r"move (?P<dir>[NSEW]) (?P<speed>\d+)|cast (?P<spell>\d{2}) (?P<power>\d+)"
        )
        assert issubclass(rs.schema, LoomUnion)
        branches = rs.schema.loom_branches()
        assert list(branches.keys()) == ["move", "cast"]

        move_fields = branches["move"].loom_fields()
        assert "dir" in move_fields
        assert "speed" in move_fields
        assert isinstance(move_fields["dir"], Categorical)
        assert isinstance(move_fields["speed"], BitInteger)

        cast_fields = branches["cast"].loom_fields()
        assert "spell" in cast_fields
        assert "power" in cast_fields

    def test_union_default_name(self):
        rs = from_regex(r"a (?P<x>\d+)|b (?P<y>\d+)")
        assert rs.schema.__name__ == "RegexUnion"

    def test_union_custom_name(self):
        rs = from_regex(r"a (?P<x>\d+)|b (?P<y>\d+)", name="MyUnion")
        assert rs.schema.__name__ == "MyUnion"

    def test_branch_name_fallback(self):
        """Branches without a clear literal prefix get numbered names."""
        rs = from_regex(r"(?P<x>\d+)|(?P<y>\d+)")
        branches = rs.schema.loom_branches()
        assert list(branches.keys()) == ["branch_0", "branch_1"]

    def test_union_vocabs_merged(self):
        rs = from_regex(
            r"move (?P<dir>[NSEW])|cast (?P<spell>fire|ice|bolt)"
        )
        assert "dir" in rs.vocabularies
        assert "spell" in rs.vocabularies


# ======================================================================
# parse() round-trips
# ======================================================================

class TestParse:
    def test_parse_categorical(self):
        rs = from_regex(r"(?P<color>red|green|blue)")
        assert rs.parse("red") == {"color": 0}
        assert rs.parse("green") == {"color": 1}
        assert rs.parse("blue") == {"color": 2}

    def test_parse_char_class(self):
        rs = from_regex(r"(?P<dir>[NSEW])")
        assert rs.parse("N") == {"dir": 0}
        assert rs.parse("W") == {"dir": 3}

    def test_parse_integer(self):
        rs = from_regex(r"(?P<code>\d{3})")
        assert rs.parse("404") == {"code": 404}
        assert rs.parse("007") == {"code": 7}

    def test_parse_boolean(self):
        rs = from_regex(r"(?P<flag>true|false)")
        assert rs.parse("true") == {"flag": True}
        assert rs.parse("false") == {"flag": False}

    def test_parse_float(self):
        rs = from_regex(r"(?P<val>\d+\.\d+)")
        assert rs.parse("3.14") == {"val": 3.14}

    def test_parse_multiple_fields(self):
        rs = from_regex(
            r"(?P<level>DEBUG|INFO|WARN|ERROR) (?P<code>\d{3}) (?P<latency>\d+\.\d+)"
        )
        result = rs.parse("WARN 404 12.5")
        assert result == {"level": 2, "code": 404, "latency": 12.5}

    def test_parse_union_branch(self):
        rs = from_regex(
            r"move (?P<dir>[NSEW]) (?P<speed>\d+)|cast (?P<spell>\d{2}) (?P<power>\d+)"
        )
        result = rs.parse("move N 50")
        assert result == {"dir": 0, "speed": 50}

        result = rs.parse("cast 03 100")
        assert result == {"spell": 3, "power": 100}

    def test_parse_no_match_raises(self):
        rs = from_regex(r"(?P<x>a|b)")
        with pytest.raises(ValueError, match="does not match"):
            rs.parse("c")


# ======================================================================
# Compiler round-trip
# ======================================================================

class TestCompilerRoundTrip:
    def test_flat_model_forward_decode(self):
        rs = from_regex(
            r"(?P<level>DEBUG|INFO|WARN|ERROR) (?P<code>\d{3})",
            name="LogEntry",
        )
        head = LoomCompiler.build_head(rs.schema, d_model=64)
        expected = 4 + math.ceil(math.log2(1000))
        assert head.total_logits == expected

        z = head(torch.randn(3, 64))
        assert z.shape == (3, expected)

        decoded = head.decode(z)
        assert "level" in decoded
        assert "code" in decoded

    def test_union_forward_decode(self):
        rs = from_regex(
            r"move (?P<dir>[NSEW]) (?P<speed>\d+)|cast (?P<spell>\d{2})"
        )
        head = LoomCompiler.build_head(rs.schema, d_model=64)

        z = head(torch.randn(4, 64))
        decoded = head.decode(z)
        assert "__opcode__" in decoded

    def test_loss_computation(self):
        rs = from_regex(r"(?P<flag>true|false) (?P<val>\d+\.\d+)")
        head = LoomCompiler.build_head(rs.schema, d_model=32)
        z = head(torch.randn(1, 32))
        targets = {
            "flag": torch.tensor([1.0]),
            "val": torch.tensor([3.5]),
        }
        total, breakdown = head.loss(z, targets)
        assert total.item() > 0
        assert "flag" in breakdown
        assert "val" in breakdown


# ======================================================================
# Error handling
# ======================================================================

class TestErrors:
    def test_no_named_groups(self):
        with pytest.raises(ValueError, match="no named groups"):
            from_regex(r"[a-z]+")

    def test_unsupported_subpattern(self):
        with pytest.raises(TypeError, match="not a recognised"):
            from_regex(r"(?P<x>[a-z]+)")

    def test_non_literal_branch_arm(self):
        with pytest.raises(TypeError, match="not a pure literal"):
            from_regex(r"(?P<x>foo\d|bar)")

    def test_empty_branch_in_union(self):
        with pytest.raises(ValueError, match="no named groups"):
            from_regex(r"hello|(?P<x>\d+)")


# ======================================================================
# RegexSchema dataclass
# ======================================================================

class TestRegexSchemaDataclass:
    def test_is_frozen(self):
        rs = from_regex(r"(?P<x>a|b)")
        with pytest.raises(AttributeError):
            rs.schema = None  # type: ignore[misc]

    def test_pattern_is_compiled(self):
        import re
        rs = from_regex(r"(?P<x>a|b)")
        assert isinstance(rs.pattern, re.Pattern)

    def test_repr_hides_field_types(self):
        rs = from_regex(r"(?P<x>a|b)")
        r = repr(rs)
        assert "_field_types" not in r
