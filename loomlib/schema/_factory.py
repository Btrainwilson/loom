"""Shared helpers for dynamically creating LoomModel / LoomUnion subclasses."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .model import LoomModel
from .union import LoomUnion
from ..types.base import LoomType


def make_loom_model(
    name: str,
    fields: OrderedDict[str, LoomType],
) -> type[LoomModel]:
    """Dynamically create a ``LoomModel`` subclass with the given fields."""
    ns: dict[str, Any] = {"__annotations__": dict(fields)}
    return type(name, (LoomModel,), ns)


def make_loom_union(
    name: str,
    branches: OrderedDict[str, type[LoomModel]],
) -> type[LoomUnion]:
    """Dynamically create a ``LoomUnion`` subclass with the given branches."""
    ns: dict[str, Any] = {"__annotations__": dict(branches)}
    return type(name, (LoomUnion,), ns)
