from __future__ import annotations

import typing
from collections import OrderedDict
from typing import Callable

from ..types.base import LoomType

_NATIVE_TYPE_MAP: dict[type, Callable[[], LoomType]] = {}


def _init_native_type_map() -> None:
    """Populate the native-type map on first access (avoids circular imports)."""
    if _NATIVE_TYPE_MAP:
        return
    from ..types.boolean import Boolean
    from ..types.bit_integer import BitInteger
    from ..types.scalar import Scalar

    _NATIVE_TYPE_MAP.update({
        bool:  lambda: Boolean(),
        int:   lambda: BitInteger(32),
        float: lambda: Scalar(),
    })


def _resolve_field_type(annotation: typing.Any) -> LoomType:
    """Turn a type annotation into a LoomType instance.

    Resolution order:
    1. Already a LoomType instance (e.g. ``Categorical(4)`` or
       ``Categorical[4]`` which returns one via ``__class_getitem__``).
    2. A native Python type with a default mapping (``bool``, ``int``,
       ``float``).
    3. Otherwise raise ``TypeError``.
    """
    if isinstance(annotation, LoomType):
        return annotation

    _init_native_type_map()
    if annotation in _NATIVE_TYPE_MAP:
        return _NATIVE_TYPE_MAP[annotation]()

    raise TypeError(
        f"Cannot resolve annotation {annotation!r} to a LoomType.  "
        f"Use a parameterised type like Categorical[4], an instance, "
        f"or a native Python type (bool, int, float)."
    )


class LoomModelMeta(type):
    """Metaclass that collects typed field annotations into an ordered dict."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, typing.Any],
    ) -> LoomModelMeta:
        cls = super().__new__(mcs, name, bases, namespace)

        if name == "LoomModel":
            return cls

        fields: OrderedDict[str, LoomType] = OrderedDict()
        annotations = typing.get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
        for field_name, annotation in annotations.items():
            if field_name.startswith("_"):
                continue
            fields[field_name] = _resolve_field_type(annotation)

        cls._loom_fields = fields  # type: ignore[attr-defined]
        return cls


class LoomModel(metaclass=LoomModelMeta):
    """Base class for flat typed structs.

    Subclass and annotate fields with Loom type descriptors::

        class MoveAction(LoomModel):
            direction: Categorical[4]
            speed: ContinuousScalar[0.0, 10.0]

    The metaclass introspects ``__annotations__`` and stores an ordered
    mapping from field name to LoomType instance in ``_loom_fields``.
    """

    _loom_fields: typing.ClassVar[OrderedDict[str, LoomType]]

    @classmethod
    def loom_fields(cls) -> OrderedDict[str, LoomType]:
        return cls._loom_fields

    @classmethod
    def total_logits(cls) -> int:
        return sum(t.logit_size() for t in cls._loom_fields.values())
