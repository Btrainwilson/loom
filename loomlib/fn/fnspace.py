from __future__ import annotations

import inspect
from typing import Any, Callable

import torch

from ..slice.allocation import SliceAllocation, SliceEntry
from ..types.base import LoomType


def _get_args(fn: Callable) -> list[str]:
    """Extract parameter names from a callable."""
    try:
        return list(inspect.signature(fn).parameters.keys())
    except (ValueError, TypeError):
        return list(fn.__code__.co_varnames[:fn.__code__.co_argcount])


class FnSpace:
    """Fn Layer: composes decoded typed values into function calls.

    Given a ``SliceAllocation`` and a set of registered functions, decodes
    all type slots and evaluates the function graph in registration order.
    """

    def __init__(self) -> None:
        self._fns: dict[str, Callable] = {}
        self._eval_order: list[str] = []

    def add_fn(self, name: str, fn: Callable) -> None:
        if name in self._fns:
            raise ValueError(f"Function '{name}' already registered.")
        self._fns[name] = fn
        self._eval_order.append(name)

    def evaluate(
        self,
        decoded_values: dict[str, Any],
        fn_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate registered functions over already-decoded values.

        *decoded_values* maps field names to their decoded tensors/values.
        Returns a dict combining the decoded values and function outputs.
        """
        results = dict(decoded_values)
        names = fn_names or self._eval_order
        for fname in names:
            fn = self._fns[fname]
            args = {arg: results[arg] for arg in _get_args(fn)}
            results[fname] = fn(**args)
        return results

    @property
    def fn_names(self) -> list[str]:
        return list(self._eval_order)
