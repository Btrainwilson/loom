"""Compile Pydantic BaseModel classes into Loom schemas.

Provides :func:`from_pydantic`, which recursively walks a (possibly nested)
Pydantic ``BaseModel`` and emits an equivalent ``LoomModel`` or ``LoomUnion``
subclass, ready to pass to ``LoomCompiler.build_head()``.

Requires ``pydantic>=2.0``.  Install with::

    pip install loomlib[pydantic]
"""

from __future__ import annotations

import enum
import re
import typing
from collections import OrderedDict
from typing import Any, Union

try:
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "pydantic>=2.0 is required for loomlib.compat.pydantic.  "
        "Install it with:  pip install loomlib[pydantic]"
    ) from exc

from ..schema.model import LoomModel
from ..schema.union import LoomUnion
from ..schema._factory import make_loom_model, make_loom_union
from ..types.base import LoomType
from ..types.boolean import Boolean
from ..types.bit_integer import BitInteger
from ..types.scalar import Scalar
from ..types.categorical import Categorical


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def from_pydantic(
    model_cls: type[BaseModel],
    *,
    name: str | None = None,
) -> type[LoomModel] | type[LoomUnion]:
    """Convert a Pydantic ``BaseModel`` into a ``LoomModel`` or ``LoomUnion``.

    Nested ``BaseModel`` fields are flattened with dot-separated names.
    A field annotated as ``Union[A, B, ...]`` where every member is a
    ``BaseModel`` subclass produces a ``LoomUnion``.

    Use ``typing.Annotated`` to override the default type mapping::

        class Sensor(BaseModel):
            temperature: Annotated[float, ContinuousScalar(-40, 85)]

    Args:
        model_cls: A pydantic ``BaseModel`` subclass.
        name: Optional name for the generated Loom schema class.
              Defaults to the Pydantic model's class name.

    Returns:
        A dynamically-created ``LoomModel`` or ``LoomUnion`` subclass.
    """
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError(
            f"Expected a pydantic BaseModel subclass, got {model_cls!r}"
        )

    model_name = name or model_cls.__name__
    hints = _model_hints(model_cls)

    # Detect Union[BaseModel, ...] fields --------------------------------
    union_fields: list[tuple[str, tuple[type[BaseModel], ...]]] = []
    for fname, annotation in hints.items():
        base = _unwrap_annotated(annotation)
        variants = _basemodel_union_variants(base)
        if variants is not None:
            union_fields.append((fname, variants))

    if len(union_fields) > 1:
        raise TypeError(
            "At most one Union[BaseModel, ...] field is supported per model; "
            f"found {len(union_fields)}: {[f for f, _ in union_fields]}"
        )

    if union_fields:
        union_fname, variants = union_fields[0]
        common_hints = {
            k: v for k, v in hints.items() if k != union_fname
        }
        common = _collect_fields_from_hints(common_hints, prefix="")
        branches: OrderedDict[str, type[LoomModel]] = OrderedDict()
        for variant_cls in variants:
            branch_name = _to_snake_case(variant_cls.__name__)
            variant_fields = _collect_fields(
                variant_cls, prefix="", skip_discriminators=True,
            )
            merged: OrderedDict[str, LoomType] = OrderedDict()
            merged.update(common)
            merged.update(variant_fields)
            branches[branch_name] = _make_loom_model(
                f"_{model_name}_{variant_cls.__name__}", merged,
            )
        return _make_loom_union(model_name, branches)

    # No union fields -> flat / nested LoomModel -------------------------
    fields = _collect_fields(model_cls, prefix="")
    return _make_loom_model(model_name, fields)


# ------------------------------------------------------------------
# Annotation introspection helpers
# ------------------------------------------------------------------

def _model_hints(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return user-defined field hints for *model_cls*, preserving ``Annotated``."""
    all_hints = typing.get_type_hints(model_cls, include_extras=True)
    field_names = set(model_cls.model_fields.keys())
    return {k: v for k, v in all_hints.items() if k in field_names}


def _unwrap_annotated(annotation: Any) -> Any:
    """Strip ``Annotated[X, ...]`` wrapper, returning the base type."""
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]
    return annotation


def _extract_thunk_override(annotation: Any) -> LoomType | None:
    """Return the first ``LoomType`` found in ``Annotated`` metadata, if any."""
    if typing.get_origin(annotation) is typing.Annotated:
        for arg in typing.get_args(annotation)[1:]:
            if isinstance(arg, LoomType):
                return arg
    return None


def _basemodel_union_variants(
    tp: Any,
) -> tuple[type[BaseModel], ...] | None:
    """If *tp* is ``Union[A, B, ...]`` with all ``BaseModel`` members, return them.

    ``None`` variants (from ``Optional``) are ignored; if only one non-None
    variant remains the result is ``None`` (that's an Optional, not a union).
    """
    origin = typing.get_origin(tp)
    if origin is not Union:
        return None
    args = typing.get_args(tp)
    non_none = tuple(a for a in args if a is not type(None))
    if len(non_none) < 2:
        return None
    if all(isinstance(a, type) and issubclass(a, BaseModel) for a in non_none):
        return non_none
    return None


def _is_single_literal(annotation: Any) -> bool:
    """True when *annotation* is ``Literal[x]`` with exactly one value."""
    base = _unwrap_annotated(annotation)
    if typing.get_origin(base) is typing.Literal:
        return len(typing.get_args(base)) == 1
    return False


def _to_snake_case(name: str) -> str:
    """Convert ``CamelCase`` to ``snake_case``."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# ------------------------------------------------------------------
# Type resolution
# ------------------------------------------------------------------

def _resolve_type(annotation: Any, field_name: str) -> LoomType:
    """Map a single (non-model, non-union) Pydantic annotation to a ``LoomType``."""
    thunk = _extract_thunk_override(annotation)
    if thunk is not None:
        return thunk

    base = _unwrap_annotated(annotation)

    # Literal[...] -> Categorical(n)
    if typing.get_origin(base) is typing.Literal:
        return Categorical(len(typing.get_args(base)))

    # Enum -> Categorical(len(members))
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return Categorical(len(base))

    # Optional[primitive] -> unwrap and resolve the inner type
    if typing.get_origin(base) is Union:
        args = typing.get_args(base)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0], field_name)

    # Primitive scalars
    if base is bool:
        return Boolean()
    if base is int:
        return BitInteger(32)
    if base is float:
        return Scalar()

    raise TypeError(
        f"Cannot convert Pydantic field '{field_name}' with type "
        f"{annotation!r} to a LoomType.  Use typing.Annotated to provide "
        f"an explicit override, e.g. Annotated[..., Categorical(10)]."
    )


# ------------------------------------------------------------------
# Field collection (recursive flattening)
# ------------------------------------------------------------------

def _collect_fields(
    model_cls: type[BaseModel],
    prefix: str,
    *,
    skip_discriminators: bool = False,
) -> OrderedDict[str, LoomType]:
    """Recursively collect and flatten fields from a Pydantic model."""
    hints = _model_hints(model_cls)
    return _collect_fields_from_hints(
        hints, prefix=prefix, skip_discriminators=skip_discriminators,
    )


def _collect_fields_from_hints(
    hints: dict[str, Any],
    prefix: str,
    *,
    skip_discriminators: bool = False,
) -> OrderedDict[str, LoomType]:
    """Collect ``LoomType`` fields from a raw dict of type hints."""
    fields: OrderedDict[str, LoomType] = OrderedDict()

    for field_name, annotation in hints.items():
        qualified = f"{prefix}.{field_name}" if prefix else field_name

        if skip_discriminators and _is_single_literal(annotation):
            continue

        # Annotated[..., LoomType()] override takes priority over everything
        thunk = _extract_thunk_override(annotation)
        if thunk is not None:
            fields[qualified] = thunk
            continue

        base = _unwrap_annotated(annotation)

        # Nested BaseModel -> recurse and flatten
        if isinstance(base, type) and issubclass(base, BaseModel):
            nested = _collect_fields(base, prefix=qualified)
            fields.update(nested)
            continue

        # Optional[BaseModel] -> presence boolean + flattened sub-fields
        if typing.get_origin(base) is Union:
            args = typing.get_args(base)
            non_none = [a for a in args if a is not type(None)]
            if type(None) in args and len(non_none) == 1:
                inner = non_none[0]
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    fields[f"{qualified}.__present__"] = Boolean()
                    nested = _collect_fields(inner, prefix=qualified)
                    fields.update(nested)
                    continue

        # Leaf type (primitive, enum, literal, etc.)
        fields[qualified] = _resolve_type(annotation, qualified)

    return fields


# ------------------------------------------------------------------
# Dynamic class construction (delegated to shared factory)
# ------------------------------------------------------------------

_make_loom_model = make_loom_model
_make_loom_union = make_loom_union
