from __future__ import annotations

import typing
from collections import OrderedDict

from .model import LoomModel


class LoomUnionMeta(type):
    """Metaclass that collects LoomModel branches into a tagged union."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, typing.Any],
    ) -> LoomUnionMeta:
        cls = super().__new__(mcs, name, bases, namespace)

        if name == "LoomUnion":
            return cls

        branches: OrderedDict[str, type[LoomModel]] = OrderedDict()
        annotations = typing.get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
        for branch_name, branch_type in annotations.items():
            if branch_name.startswith("_"):
                continue
            if not (isinstance(branch_type, type) and issubclass(branch_type, LoomModel)):
                raise TypeError(
                    f"LoomUnion branch '{branch_name}' must be a LoomModel subclass, "
                    f"got {branch_type!r}"
                )
            branches[branch_name] = branch_type

        cls._loom_branches = branches  # type: ignore[attr-defined]
        return cls


class LoomUnion(metaclass=LoomUnionMeta):
    """Tagged union over LoomModel branches.

    Subclass and annotate each branch::

        class AgentAction(LoomUnion):
            move: MoveAction
            cast: CastSpell

    The compiler allocates an opcode slice (sized by the number of branches)
    followed by each branch's fields sequentially.
    """

    _loom_branches: typing.ClassVar[OrderedDict[str, type[LoomModel]]]

    @classmethod
    def loom_branches(cls) -> OrderedDict[str, type[LoomModel]]:
        return cls._loom_branches

    @classmethod
    def num_branches(cls) -> int:
        return len(cls._loom_branches)

    @classmethod
    def branch_names(cls) -> list[str]:
        return list(cls._loom_branches.keys())
