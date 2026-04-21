"""Convert regular expressions into Loom schemas.

Provides :func:`from_regex`, which parses a regex pattern string and emits an
equivalent ``LoomModel`` or ``LoomUnion`` subclass bundled with vocabulary
metadata and a :meth:`~RegexSchema.parse` helper for converting matched
strings into typed dicts.

Named groups (``(?P<name>...)``) become fields.  Top-level alternation (``|``)
produces a ``LoomUnion`` whose branches are derived from each alternative.
"""

from __future__ import annotations

import math
import re
import re._constants as _c
import re._parser as _parser
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

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

@dataclass(frozen=True)
class RegexSchema:
    """Result of :func:`from_regex`.

    Bundles a ``LoomModel`` or ``LoomUnion`` class with the metadata needed
    to parse strings that match the original regex.
    """

    schema: type[LoomModel] | type[LoomUnion]
    vocabularies: dict[str, list[str]]
    pattern: re.Pattern[str]
    _field_types: dict[str, LoomType] = field(repr=False)

    def parse(self, string: str) -> dict[str, Any]:
        """Match *string* against the pattern and return typed field values.

        Raises :class:`ValueError` if the string does not match.
        """
        m = self.pattern.fullmatch(string)
        if m is None:
            raise ValueError(
                f"String {string!r} does not match pattern {self.pattern.pattern!r}"
            )
        result: dict[str, Any] = {}
        for name, raw in m.groupdict().items():
            if raw is None:
                continue
            result[name] = self._convert_value(name, raw)
        return result

    def _convert_value(self, name: str, raw: str) -> Any:
        loom_type = self._field_types[name]
        if isinstance(loom_type, Boolean):
            return _parse_boolean(raw)
        if isinstance(loom_type, Categorical):
            vocab = self.vocabularies.get(name)
            if vocab is not None:
                return vocab.index(raw)
            return int(raw)
        if isinstance(loom_type, BitInteger):
            return int(raw)
        if isinstance(loom_type, Scalar):
            return float(raw)
        raise TypeError(f"Cannot convert field {name!r} with type {loom_type!r}")


_BOOL_TRUE = frozenset({"true", "True", "TRUE", "yes", "Yes", "YES", "1"})
_BOOL_FALSE = frozenset({"false", "False", "FALSE", "no", "No", "NO", "0"})


def _parse_boolean(raw: str) -> bool:
    if raw in _BOOL_TRUE:
        return True
    if raw in _BOOL_FALSE:
        return False
    raise ValueError(f"Cannot interpret {raw!r} as boolean")


def from_regex(
    pattern: str,
    *,
    name: str | None = None,
) -> RegexSchema:
    """Convert a regex pattern string into a ``LoomModel`` or ``LoomUnion``.

    Named groups (``(?P<name>...)``) become fields.  The sub-pattern inside
    each group is analysed to select an appropriate :class:`~loomlib.LoomType`.

    Top-level alternation (``|``) produces a ``LoomUnion`` whose branch names
    are derived from leading literal prefixes when possible.

    Args:
        pattern: A regular expression string.
        name: Optional name for the generated schema class.  Defaults to
              ``"RegexModel"`` (or ``"RegexUnion"``).

    Returns:
        A :class:`RegexSchema` bundling the schema, vocabularies, and a
        compiled pattern.
    """
    parsed = _parser.parse(pattern)
    group_names: dict[int, str] = {
        v: k for k, v in parsed.state.groupdict.items()
    }
    compiled = re.compile(pattern)

    top_nodes = list(parsed)

    if _is_top_level_branch(top_nodes):
        return _build_union(top_nodes, group_names, compiled, name)
    return _build_flat(top_nodes, group_names, compiled, name)


# ------------------------------------------------------------------
# Top-level structure detection
# ------------------------------------------------------------------

def _is_top_level_branch(nodes: list) -> bool:
    """True when the pattern is a top-level alternation (``|``)."""
    return len(nodes) == 1 and nodes[0][0] == _c.BRANCH


# ------------------------------------------------------------------
# Flat model (no top-level alternation)
# ------------------------------------------------------------------

def _build_flat(
    nodes: list,
    group_names: dict[int, str],
    compiled: re.Pattern[str],
    name: str | None,
) -> RegexSchema:
    fields, vocabs, ftypes = _extract_fields(nodes, group_names)
    if not fields:
        raise ValueError("Pattern contains no named groups — cannot build a LoomModel")
    schema_name = name or "RegexModel"
    schema = make_loom_model(schema_name, fields)
    return RegexSchema(
        schema=schema,
        vocabularies=vocabs,
        pattern=compiled,
        _field_types=ftypes,
    )


# ------------------------------------------------------------------
# Union (top-level alternation)
# ------------------------------------------------------------------

def _build_union(
    nodes: list,
    group_names: dict[int, str],
    compiled: re.Pattern[str],
    name: str | None,
) -> RegexSchema:
    _, (_, branches_ast) = nodes[0]
    branch_models: OrderedDict[str, type[LoomModel]] = OrderedDict()
    all_vocabs: dict[str, list[str]] = {}
    all_ftypes: dict[str, LoomType] = {}

    for i, branch_nodes in enumerate(branches_ast):
        branch_name = _infer_branch_name(branch_nodes, i)
        fields, vocabs, ftypes = _extract_fields(branch_nodes, group_names)
        if not fields:
            raise ValueError(
                f"Branch {i} of top-level alternation contains no named groups"
            )
        branch_models[branch_name] = make_loom_model(
            f"_{name or 'RegexUnion'}_{branch_name}", fields,
        )
        all_vocabs.update(vocabs)
        all_ftypes.update(ftypes)

    schema_name = name or "RegexUnion"
    schema = make_loom_union(schema_name, branch_models)
    return RegexSchema(
        schema=schema,
        vocabularies=all_vocabs,
        pattern=compiled,
        _field_types=all_ftypes,
    )


def _infer_branch_name(branch_nodes: list, index: int) -> str:
    """Derive a branch name from a leading literal prefix, or fall back."""
    chars: list[str] = []
    for op, av in branch_nodes:
        if op == _c.LITERAL:
            chars.append(chr(av))
        else:
            break
    prefix = "".join(chars).strip()
    if prefix:
        ident = re.sub(r"[^a-zA-Z0-9]+", "_", prefix).strip("_").lower()
        if ident and ident[0].isalpha():
            return ident
    return f"branch_{index}"


# ------------------------------------------------------------------
# Field extraction from a node list
# ------------------------------------------------------------------

def _extract_fields(
    nodes: list,
    group_names: dict[int, str],
) -> tuple[OrderedDict[str, LoomType], dict[str, list[str]], dict[str, LoomType]]:
    """Walk *nodes* and collect named-group fields.

    Returns ``(fields, vocabularies, field_types)`` where *fields* maps
    field names to ``LoomType`` instances, *vocabularies* maps categorical
    field names to their ordered labels, and *field_types* is a flat
    name-to-type dict used by ``RegexSchema.parse``.
    """
    fields: OrderedDict[str, LoomType] = OrderedDict()
    vocabs: dict[str, list[str]] = {}
    ftypes: dict[str, LoomType] = {}

    for op, av in nodes:
        if op == _c.SUBPATTERN:
            group_id, _, _, sub_nodes = av
            fname = group_names.get(group_id)
            if fname is None:
                continue
            loom_type, vocab = _classify_subpattern(sub_nodes, fname)
            fields[fname] = loom_type
            ftypes[fname] = loom_type
            if vocab is not None:
                vocabs[fname] = vocab
    return fields, vocabs, ftypes


# ------------------------------------------------------------------
# Sub-pattern classification
# ------------------------------------------------------------------

def _classify_subpattern(
    nodes: list,
    field_name: str,
) -> tuple[LoomType, list[str] | None]:
    """Determine the ``LoomType`` for the content of a named group.

    Returns ``(loom_type, vocabulary)`` where *vocabulary* is a list of
    string labels for ``Categorical`` types, or ``None`` otherwise.
    """
    if len(nodes) == 1:
        op, av = nodes[0]

        # BRANCH: literal alternation  (foo|bar|baz)
        if op == _c.BRANCH:
            return _classify_branch(av, field_name)

        # IN: character class  [NSEW]
        if op == _c.IN:
            return _classify_char_class(av, field_name)

        # MAX_REPEAT / MIN_REPEAT over a single element
        if op in (_c.MAX_REPEAT, _c.MIN_REPEAT):
            return _classify_repeat(av, field_name)

    # Multi-node: look for float-like  \d+\.\d+
    float_type = _try_classify_float(nodes, field_name)
    if float_type is not None:
        return float_type, None

    # Single literal digit category  \d  (parsed as IN with CATEGORY_DIGIT)
    if len(nodes) == 1 and nodes[0][0] == _c.IN:
        return _classify_char_class(nodes[0][1], field_name)

    raise TypeError(
        f"Cannot classify sub-pattern for named group {field_name!r}.  "
        f"Use a pattern that resolves to a literal alternation, character "
        f"class, digit sequence, or float."
    )


# -- Branch (alternation) --------------------------------------------------

def _classify_branch(
    av: tuple,
    field_name: str,
) -> tuple[LoomType, list[str] | None]:
    """Classify ``(a|b|c)`` inside a named group."""
    _, arms = av
    labels: list[str] = []
    for arm in arms:
        lit = _literal_string(arm)
        if lit is None:
            raise TypeError(
                f"Branch arm in group {field_name!r} is not a pure literal "
                f"sequence — cannot convert to Categorical."
            )
        labels.append(lit)

    # Boolean special case
    label_set = frozenset(labels)
    if label_set <= (_BOOL_TRUE | _BOOL_FALSE) and len(labels) == 2:
        return Boolean(), None

    return Categorical(len(labels)), labels


# -- Character class -------------------------------------------------------

def _classify_char_class(
    members: list,
    field_name: str,
) -> tuple[LoomType, list[str] | None]:
    """Classify ``[...]`` inside a named group."""
    chars = _enumerate_class_members(members)
    if chars is not None:
        char_set = frozenset(chars)
        if len(chars) == 2 and char_set <= (_BOOL_TRUE | _BOOL_FALSE):
            return Boolean(), None
        if len(chars) == 10 and chars == list("0123456789"):
            return Categorical(10), [str(i) for i in range(10)]
        return Categorical(len(chars)), chars

    if _is_digit_category(members):
        return Categorical(10), [str(i) for i in range(10)]

    raise TypeError(
        f"Character class in group {field_name!r} contains elements that "
        f"cannot be enumerated into a Categorical."
    )


def _enumerate_class_members(members: list) -> list[str] | None:
    """Return an ordered list of characters in a class, or None if not enumerable."""
    chars: list[str] = []
    for op, av in members:
        if op == _c.LITERAL:
            chars.append(chr(av))
        elif op == _c.RANGE:
            lo, hi = av
            if hi - lo > 256:
                return None
            chars.extend(chr(c) for c in range(lo, hi + 1))
        elif op == _c.CATEGORY:
            if av == _c.CATEGORY_DIGIT:
                chars.extend(str(i) for i in range(10))
            else:
                return None
        elif op == _c.NEGATE:
            return None
        else:
            return None
    return chars


def _is_digit_category(members: list) -> bool:
    """True when the class is exactly ``\\d`` or ``[0-9]``."""
    if len(members) == 1:
        op, av = members[0]
        if op == _c.CATEGORY and av == _c.CATEGORY_DIGIT:
            return True
        if op == _c.RANGE and av == (48, 57):  # ord('0'), ord('9')
            return True
    return False


# -- Repetition ------------------------------------------------------------

def _classify_repeat(
    av: tuple,
    field_name: str,
) -> tuple[LoomType, list[str] | None]:
    """Classify ``\\d{k}``, ``\\d+``, ``\\d{m,n}`` and similar repeats."""
    min_count, max_count, sub_nodes = av

    if len(sub_nodes) == 1 and sub_nodes[0][0] == _c.IN:
        class_members = sub_nodes[0][1]
        if _is_digit_category(class_members):
            return _digit_repeat_to_type(min_count, max_count, field_name)

    raise TypeError(
        f"Repeated sub-pattern in group {field_name!r} is not a recognised "
        f"digit pattern — cannot convert to a LoomType."
    )


_MAXREPEAT = _c.MAXREPEAT


def _digit_repeat_to_type(
    min_count: int,
    max_count: int,
    field_name: str,
) -> tuple[LoomType, list[str] | None]:
    """Map a digit repetition to a ``BitInteger``."""
    if max_count == _MAXREPEAT:
        return BitInteger(32), None

    max_value = 10**max_count - 1
    bits = max(1, math.ceil(math.log2(max_value + 1)))
    return BitInteger(bits), None


# -- Float pattern ---------------------------------------------------------

def _try_classify_float(
    nodes: list,
    field_name: str,
) -> LoomType | None:
    """Recognise ``\\d+\\.\\d+`` or ``\\d{m,n}\\.\\d{m,n}`` as a float."""
    # Expect: digit_part, LITERAL('.'), digit_part
    if len(nodes) != 3:
        return None

    op0, _ = nodes[0]
    op1, av1 = nodes[1]
    op2, _ = nodes[2]

    if op1 != _c.LITERAL or av1 != ord("."):
        return None

    if not _is_digit_node(nodes[0]) or not _is_digit_node(nodes[2]):
        return None

    return Scalar()


def _is_digit_node(node: tuple) -> bool:
    """True when *node* is a single digit or repeated digit pattern."""
    op, av = node
    if op == _c.IN:
        return _is_digit_category(av)
    if op in (_c.MAX_REPEAT, _c.MIN_REPEAT):
        _, _, sub = av
        if len(sub) == 1 and sub[0][0] == _c.IN:
            return _is_digit_category(sub[0][1])
    return False


# -- Utility ---------------------------------------------------------------

def _literal_string(arm: list) -> str | None:
    """Extract a pure literal string from an AST arm, or return None."""
    chars: list[str] = []
    for op, av in arm:
        if op == _c.LITERAL:
            chars.append(chr(av))
        else:
            return None
    return "".join(chars)
