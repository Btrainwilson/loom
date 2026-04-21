"""Tests for the Pydantic -> LoomModel/LoomUnion compiler."""

import enum
from typing import Annotated, Literal, Optional, Union

import pytest
import torch
from pydantic import BaseModel

from loomlib import (
    LoomCompiler,
    LoomModel,
    LoomUnion,
    Categorical,
    ContinuousScalar,
    BitInteger,
    Boolean,
    Scalar,
)
from loomlib.compat.pydantic import from_pydantic


# ======================================================================
# Helpers / fixtures
# ======================================================================

class Color(enum.IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


class Direction(enum.Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"


# ======================================================================
# Flat model tests
# ======================================================================

class TestFlatModel:
    def test_bool_int_float(self):
        class Sensor(BaseModel):
            active: bool
            reading: int
            voltage: float

        schema = from_pydantic(Sensor)
        assert issubclass(schema, LoomModel)
        fields = schema.loom_fields()
        assert list(fields.keys()) == ["active", "reading", "voltage"]
        assert isinstance(fields["active"], Boolean)
        assert isinstance(fields["reading"], BitInteger)
        assert fields["reading"].logit_size() == 32
        assert isinstance(fields["voltage"], Scalar)
        assert schema.total_logits() == 1 + 32 + 1

    def test_custom_name(self):
        class M(BaseModel):
            x: float

        schema = from_pydantic(M, name="MySensor")
        assert schema.__name__ == "MySensor"

    def test_default_name_from_pydantic_class(self):
        class Gadget(BaseModel):
            on: bool

        schema = from_pydantic(Gadget)
        assert schema.__name__ == "Gadget"


# ======================================================================
# Annotated override tests
# ======================================================================

class TestAnnotatedOverrides:
    def test_continuous_scalar_override(self):
        class Env(BaseModel):
            temperature: Annotated[float, ContinuousScalar(-40.0, 85.0)]

        schema = from_pydantic(Env)
        fields = schema.loom_fields()
        thunk = fields["temperature"]
        assert isinstance(thunk, ContinuousScalar)
        assert thunk.logit_size() == 1

    def test_bit_integer_override(self):
        class Pkt(BaseModel):
            port: Annotated[int, BitInteger(16)]

        schema = from_pydantic(Pkt)
        fields = schema.loom_fields()
        assert isinstance(fields["port"], BitInteger)
        assert fields["port"].logit_size() == 16

    def test_categorical_override(self):
        class Ctrl(BaseModel):
            mode: Annotated[int, Categorical(8)]

        schema = from_pydantic(Ctrl)
        fields = schema.loom_fields()
        assert isinstance(fields["mode"], Categorical)
        assert fields["mode"].logit_size() == 8


# ======================================================================
# Enum and Literal tests
# ======================================================================

class TestEnumAndLiteral:
    def test_int_enum(self):
        class Pixel(BaseModel):
            color: Color

        schema = from_pydantic(Pixel)
        fields = schema.loom_fields()
        assert isinstance(fields["color"], Categorical)
        assert fields["color"].logit_size() == 3

    def test_str_enum(self):
        class Nav(BaseModel):
            heading: Direction

        schema = from_pydantic(Nav)
        fields = schema.loom_fields()
        assert isinstance(fields["heading"], Categorical)
        assert fields["heading"].logit_size() == 4

    def test_literal_multiple_values(self):
        class Cmd(BaseModel):
            action: Literal["run", "walk", "stop"]

        schema = from_pydantic(Cmd)
        fields = schema.loom_fields()
        assert isinstance(fields["action"], Categorical)
        assert fields["action"].logit_size() == 3


# ======================================================================
# Nested model tests (flattening)
# ======================================================================

class TestNestedModels:
    def test_single_nesting(self):
        class Position(BaseModel):
            x: float
            y: float

        class Agent(BaseModel):
            health: int
            position: Position

        schema = from_pydantic(Agent)
        fields = schema.loom_fields()
        assert list(fields.keys()) == ["health", "position.x", "position.y"]
        assert isinstance(fields["health"], BitInteger)
        assert isinstance(fields["position.x"], Scalar)
        assert isinstance(fields["position.y"], Scalar)

    def test_deep_nesting(self):
        class Coord(BaseModel):
            val: float

        class Vec2(BaseModel):
            x: Coord
            y: Coord

        class Entity(BaseModel):
            pos: Vec2
            alive: bool

        schema = from_pydantic(Entity)
        fields = schema.loom_fields()
        assert list(fields.keys()) == [
            "pos.x.val", "pos.y.val", "alive",
        ]

    def test_nested_with_annotated_override(self):
        class Stats(BaseModel):
            hp: Annotated[int, BitInteger(8)]
            mp: Annotated[int, BitInteger(8)]

        class Character(BaseModel):
            stats: Stats
            level: int

        schema = from_pydantic(Character)
        fields = schema.loom_fields()
        assert fields["stats.hp"].logit_size() == 8
        assert fields["stats.mp"].logit_size() == 8
        assert fields["level"].logit_size() == 32


# ======================================================================
# Optional field tests
# ======================================================================

class TestOptionalFields:
    def test_optional_primitive(self):
        class M(BaseModel):
            score: Optional[float] = None

        schema = from_pydantic(M)
        fields = schema.loom_fields()
        assert list(fields.keys()) == ["score"]
        assert isinstance(fields["score"], Scalar)

    def test_optional_basemodel(self):
        class Weapon(BaseModel):
            damage: int
            range: float

        class Player(BaseModel):
            name_id: int
            weapon: Optional[Weapon] = None

        schema = from_pydantic(Player)
        fields = schema.loom_fields()
        expected_keys = [
            "name_id",
            "weapon.__present__",
            "weapon.damage",
            "weapon.range",
        ]
        assert list(fields.keys()) == expected_keys
        assert isinstance(fields["weapon.__present__"], Boolean)
        assert isinstance(fields["weapon.damage"], BitInteger)
        assert isinstance(fields["weapon.range"], Scalar)


# ======================================================================
# Union / LoomUnion tests
# ======================================================================

class TestUnionHandling:
    def test_basic_union(self):
        class MoveAction(BaseModel):
            direction: Literal["n", "s", "e", "w"]
            speed: float

        class CastSpell(BaseModel):
            spell_id: int
            power: float

        class GameAction(BaseModel):
            action: Union[MoveAction, CastSpell]

        schema = from_pydantic(GameAction)
        assert issubclass(schema, LoomUnion)
        branches = schema.loom_branches()
        assert list(branches.keys()) == ["move_action", "cast_spell"]

        move_fields = branches["move_action"].loom_fields()
        assert "direction" in move_fields
        assert "speed" in move_fields

        cast_fields = branches["cast_spell"].loom_fields()
        assert "spell_id" in cast_fields
        assert "power" in cast_fields

    def test_union_with_common_fields(self):
        class Attack(BaseModel):
            target: int

        class Defend(BaseModel):
            shield: bool

        class Turn(BaseModel):
            tick: int
            action: Union[Attack, Defend]

        schema = from_pydantic(Turn)
        assert issubclass(schema, LoomUnion)
        branches = schema.loom_branches()

        for branch_model in branches.values():
            assert "tick" in branch_model.loom_fields()

    def test_union_discriminator_stripping(self):
        """Single-value Literal fields (discriminators) are excluded."""

        class MoveAction(BaseModel):
            type: Literal["move"]
            dx: float
            dy: float

        class JumpAction(BaseModel):
            type: Literal["jump"]
            height: float

        class Action(BaseModel):
            action: Union[MoveAction, JumpAction]

        schema = from_pydantic(Action)
        branches = schema.loom_branches()

        move_fields = branches["move_action"].loom_fields()
        assert "type" not in move_fields
        assert "dx" in move_fields
        assert "dy" in move_fields

        jump_fields = branches["jump_action"].loom_fields()
        assert "type" not in jump_fields
        assert "height" in jump_fields


# ======================================================================
# Error handling tests
# ======================================================================

class TestErrors:
    def test_unsupported_str_type(self):
        class M(BaseModel):
            label: str

        with pytest.raises(TypeError, match="Cannot convert"):
            from_pydantic(M)

    def test_unsupported_list_type(self):
        class M(BaseModel):
            items: list[int]

        with pytest.raises(TypeError, match="Cannot convert"):
            from_pydantic(M)

    def test_non_basemodel_raises(self):
        with pytest.raises(TypeError, match="Expected a pydantic BaseModel"):
            from_pydantic(dict)  # type: ignore[arg-type]

    def test_multiple_union_fields_raises(self):
        class A(BaseModel):
            x: int

        class B(BaseModel):
            y: int

        class C(BaseModel):
            z: int

        class Bad(BaseModel):
            first: Union[A, B]
            second: Union[A, C]

        with pytest.raises(TypeError, match="At most one"):
            from_pydantic(Bad)


# ======================================================================
# Compiler round-trip tests
# ======================================================================

class TestCompilerRoundTrip:
    def test_flat_model_forward_decode(self):
        class Ctrl(BaseModel):
            direction: Annotated[int, Categorical(4)]
            speed: float
            active: bool

        schema = from_pydantic(Ctrl)
        head = LoomCompiler.build_head(schema, d_model=64)
        assert head.total_logits == 4 + 1 + 1

        z = head(torch.randn(3, 64))
        assert z.shape == (3, 6)

        decoded = head.decode(z)
        assert "direction" in decoded
        assert "speed" in decoded
        assert "active" in decoded

    def test_nested_model_forward_decode(self):
        class Pos(BaseModel):
            x: float
            y: float

        class Agent(BaseModel):
            alive: bool
            pos: Pos

        schema = from_pydantic(Agent)
        head = LoomCompiler.build_head(schema, d_model=32)
        assert head.total_logits == 1 + 1 + 1

        z = head(torch.randn(2, 32))
        decoded = head.decode(z)
        assert "alive" in decoded
        assert "pos.x" in decoded
        assert "pos.y" in decoded

    def test_union_forward_decode(self):
        class Move(BaseModel):
            type: Literal["move"]
            dx: float

        class Shoot(BaseModel):
            type: Literal["shoot"]
            target_id: Annotated[int, Categorical(8)]

        class Action(BaseModel):
            action: Union[Move, Shoot]

        schema = from_pydantic(Action)
        head = LoomCompiler.build_head(schema, d_model=64)

        z = head(torch.randn(4, 64))
        decoded = head.decode(z)
        assert "__opcode__" in decoded

    def test_loss_computation(self):
        class Obs(BaseModel):
            flag: bool
            value: float

        schema = from_pydantic(Obs)
        head = LoomCompiler.build_head(schema, d_model=32)
        z = head(torch.randn(1, 32))
        targets = {
            "flag": torch.tensor([1.0]),
            "value": torch.tensor([3.5]),
        }
        total, breakdown = head.loss(z, targets)
        assert total.item() > 0
        assert "flag" in breakdown
        assert "value" in breakdown

    def test_enum_compile_and_decode(self):
        class Pixel(BaseModel):
            color: Color
            brightness: float

        schema = from_pydantic(Pixel)
        head = LoomCompiler.build_head(schema, d_model=32)
        assert head.total_logits == 3 + 1

        z = head(torch.randn(5, 32))
        decoded = head.decode(z)
        assert "color" in decoded
        assert "brightness" in decoded
